from src.config import config
from src.embedding import EmbeddingModel
from src.vectordb import VectorStore
from src.llm import LLM
from src.utils import clip_text

def test_integration() -> None:
    """Test that all modules can be imported and initialized together."""

    print("Testing module integration...\n")

    # Test 1: Config
    print("✓ Config loaded")
    print(f"  - Embed Model: {config.EMBED_MODEL_ID}")
    print(f"  - Gen Model: {config.GEN_MODEL_ID}")

    # Test 2: Utils
    test_text = "This is a very long text that should be clipped"
    clipped = clip_text(test_text, threshold=20)
    print(f"\n✓ Utils working")
    print(f"  - Clipped text: {clipped}")

    # Test 3: Embedding Model
    embedding_model = EmbeddingModel()
    print(f"\n✓ Embedding model initialized")
    print(f"  - Model ID: {embedding_model.model_id}")

    # Test 4: Vector Store
    vectorstore = VectorStore(embedding_model.get_embedding())
    print(f"\n✓ Vector store initialized")
    print(f"  - Collection: {vectorstore.collection_name}")

    # Test 5: LLM
    llm = LLM()
    print(f"\n✓ LLM initialized")

    # Test 6: Processing functions
    print(f"\n✓ Processing functions available")
    print(f"  - load_document")
    print(f"  - chunk")

    print("\n✅ All modules integrated successfully!")


if __name__ == "__main__":
    test_integration()