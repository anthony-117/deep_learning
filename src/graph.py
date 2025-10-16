from typing import TypedDict, Annotated, Literal, Optional
from operator import add

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from .vectordb import VectorStore
from .llm import LLMModel


class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    documents: list[Document]
    steps: Annotated[list[str], add]
    rewrite_count: int
    relevance_scores: list[float]
    answer_grounded: bool


class RAGGraph:
    """
    LangGraph-based RAG workflow with self-correction and query rewriting.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMModel,
        max_rewrites: int = 2,
        relevance_threshold: float = 0.5
    ):
        """
        Initialize the RAG graph.

        Args:
            vector_store: VectorStore instance for document retrieval
            llm: LLM instance for generation
            max_rewrites: Maximum number of query rewrites allowed
            relevance_threshold: Minimum relevance score for documents
        """
        self.vector_store = vector_store
        self.llm = llm.get_llm()
        self.retriever = vector_store.get_retriever()
        self.max_rewrites = max_rewrites
        self.relevance_threshold = relevance_threshold

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("rewrite_query", self._rewrite_query)

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Add edges
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional edge after grading
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "end": END
            }
        )

        workflow.add_edge("generate", "check_hallucination")

        # Conditional edge after hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            self._decide_after_hallucination,
            {
                "end": END,
                "rewrite": "rewrite_query"
            }
        )

        workflow.add_edge("rewrite_query", "retrieve")

        return workflow

    def _analyze_query(self, state: GraphState) -> GraphState:
        """Analyze and potentially expand the user's query."""
        steps = [f"Analyzing query: {state['question']}"]

        # Simple query analysis - you can make this more sophisticated
        return {
            **state,
            "steps": steps,
            "rewrite_count": 0
        }

    def _retrieve_documents(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents from vector store."""
        question = state["question"]
        documents = self.retriever.invoke(question)

        steps = [f"Retrieved {len(documents)} documents"]

        return {
            **state,
            "documents": documents,
            "steps": steps
        }

    def _grade_documents(self, state: GraphState) -> GraphState:
        """
        Grade the relevance of retrieved documents.

        Uses an LLM to determine if documents are relevant to the question.
        """
        question = state["question"]
        documents = state["documents"]

        # Grading prompt
        grade_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.

            Retrieved document:
            {document}

            User question: {question}

            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

            Provide only 'yes' or 'no' as output."""
        )

        chain = grade_prompt | self.llm | StrOutputParser()

        # Score each document
        relevance_scores = []
        filtered_docs = []

        for doc in documents:
            score = chain.invoke({
                "document": doc.page_content,
                "question": question
            })

            if "yes" in score.lower():
                relevance_scores.append(1.0)
                filtered_docs.append(doc)
            else:
                relevance_scores.append(0.0)

        steps = [f"Graded documents: {len(filtered_docs)}/{len(documents)} relevant"]

        return {
            **state,
            "documents": filtered_docs,
            "relevance_scores": relevance_scores,
            "steps": steps
        }

    def _decide_after_grading(self, state: GraphState) -> Literal["generate", "rewrite", "end"]:
        """
        Decide next step after document grading.

        Returns:
            - "generate" if we have relevant documents
            - "rewrite" if no relevant docs and rewrites available
            - "end" if no relevant docs and max rewrites reached
        """
        if state["documents"]:
            return "generate"

        if state["rewrite_count"] < self.max_rewrites:
            return "rewrite"

        return "end"

    def _generate_answer(self, state: GraphState) -> GraphState:
        """Generate answer using retrieved documents."""
        question = state["question"]
        documents = state["documents"]

        # Format documents
        context = "\n\n".join([doc.page_content for doc in documents])

        # Generation prompt
        gen_prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.

            Question: {question}

            Context: {context}

            Answer:"""
        )

        chain = gen_prompt | self.llm | StrOutputParser()

        generation = chain.invoke({
            "question": question,
            "context": context
        })

        steps = ["Generated answer from documents"]

        return {
            **state,
            "generation": generation,
            "steps": steps
        }

    def _check_hallucination(self, state: GraphState) -> GraphState:
        """
        Check if the generated answer is grounded in the retrieved documents.
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        # Hallucination check prompt
        hallucination_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer is grounded in / supported by a set of facts.

            Set of facts:
            {documents}

            Answer: {generation}

            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the facts.
            Provide only 'yes' or 'no' as output."""
        )

        chain = hallucination_prompt | self.llm | StrOutputParser()

        context = "\n\n".join([doc.page_content for doc in documents])
        score = chain.invoke({
            "documents": context,
            "generation": generation
        })

        is_grounded = "yes" in score.lower()
        steps = [f"Hallucination check: {'passed' if is_grounded else 'failed'}"]

        return {
            **state,
            "answer_grounded": is_grounded,
            "steps": steps
        }

    def _decide_after_hallucination(self, state: GraphState) -> Literal["end", "rewrite"]:
        """
        Decide next step after hallucination check.

        Returns:
            - "end" if answer is grounded or max rewrites reached
            - "rewrite" if answer is hallucinated and rewrites available
        """
        if state["answer_grounded"]:
            return "end"

        if state["rewrite_count"] < self.max_rewrites:
            return "rewrite"

        return "end"

    def _rewrite_query(self, state: GraphState) -> GraphState:
        """Rewrite the query to improve retrieval."""
        question = state["question"]

        # Query rewrite prompt
        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a question re-writer that converts an input question to a better version that is optimized
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.

            Here is the initial question:
            {question}

            Formulate an improved question:"""
        )

        chain = rewrite_prompt | self.llm | StrOutputParser()

        better_question = chain.invoke({"question": question})

        steps = [f"Rewrote query (attempt {state['rewrite_count'] + 1})"]

        return {
            **state,
            "question": better_question,
            "rewrite_count": state["rewrite_count"] + 1,
            "steps": steps
        }

    def invoke(self, question: str) -> dict:
        """
        Run the RAG graph with a question.

        Args:
            question: User's question

        Returns:
            Dictionary with answer, documents, and processing steps
        """
        initial_state: GraphState = {
            "question": question,
            "generation": None,
            "documents": [],
            "steps": [],
            "rewrite_count": 0,
            "relevance_scores": [],
            "answer_grounded": False
        }

        # Run the graph
        result = self.app.invoke(initial_state)

        return {
            "question": result["question"],
            "answer": result.get("generation", "I couldn't find a relevant answer."),
            "documents": result["documents"],
            "steps": result["steps"],
            "rewrites": result["rewrite_count"],
            "grounded": result.get("answer_grounded", False)
        }

    def stream(self, question: str):
        """
        Stream the RAG graph execution step-by-step.

        Args:
            question: User's question

        Yields:
            State updates at each step of the graph
        """
        initial_state: GraphState = {
            "question": question,
            "generation": None,
            "documents": [],
            "steps": [],
            "rewrite_count": 0,
            "relevance_scores": [],
            "answer_grounded": False
        }

        for output in self.app.stream(initial_state):
            yield output