'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Brain, Search, FileText, CheckCircle, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: {
    steps?: string[]
    documents?: Array<{
      content: string
      metadata: any
    }>
    rewrites?: number
    grounded?: boolean
  }
}

interface ProcessingStep {
  id: string
  step: string
  status: 'pending' | 'active' | 'completed' | 'error'
  timestamp: number
  details?: any
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([])
  const [showSteps, setShowSteps] = useState(true)
  const [showSources, setShowSources] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, processingSteps])

  useEffect(() => {
    // Initialize WebSocket connection
    const ws = new WebSocket('ws://localhost:8000/ws')
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'step') {
        setProcessingSteps([{
          id: Date.now().toString(),
          step: data.step,
          status: 'active',
          timestamp: data.timestamp || Date.now(),
          details: data.state
        }])
      } else if (data.type === 'step_update') {
        setProcessingSteps(prev => {
          const newSteps = [...prev]
          const lastStep = newSteps[newSteps.length - 1]
          if (lastStep) {
            lastStep.status = 'completed'
            lastStep.details = data.output
          }
          
          // Add new step if there's a new node
          const outputKeys = Object.keys(data.output)
          if (outputKeys.length > 0) {
            const nodeName = outputKeys[0]
            const stepName = getStepDisplayName(nodeName)
            newSteps.push({
              id: Date.now().toString(),
              step: stepName,
              status: 'active',
              timestamp: data.timestamp || Date.now(),
              details: data.output[nodeName]
            })
          }
          
          return newSteps
        })
      } else if (data.type === 'final_result') {
        setIsLoading(false)
        setProcessingSteps([])
        
        const newMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: data.result.answer,
          timestamp: new Date(),
          metadata: {
            steps: data.result.steps,
            documents: data.result.documents,
            rewrites: data.result.rewrites,
            grounded: data.result.grounded
          }
        }
        
        setMessages(prev => [...prev, newMessage])
      } else if (data.error) {
        setIsLoading(false)
        setProcessingSteps([])
        
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: `Error: ${data.error}`,
          timestamp: new Date()
        }
        
        setMessages(prev => [...prev, errorMessage])
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    return () => {
      ws.close()
    }
  }, [])

  const getStepDisplayName = (nodeName: string): string => {
    const stepNames: { [key: string]: string } = {
      'analyze_query': '🔍 Analyzing Query',
      'retrieve': '📚 Retrieving Documents',
      'grade_documents': '📊 Grading Document Relevance',
      'generate': '🤖 Generating Answer',
      'check_hallucination': '🔍 Checking for Hallucinations',
      'rewrite_query': '✏️ Rewriting Query'
    }
    return stepNames[nodeName] || `🔄 ${nodeName}`
  }

  const getStepIcon = (step: string) => {
    if (step.includes('Analyzing')) return <Brain className="w-4 h-4" />
    if (step.includes('Retrieving')) return <Search className="w-4 h-4" />
    if (step.includes('Grading')) return <CheckCircle className="w-4 h-4" />
    if (step.includes('Generating')) return <Bot className="w-4 h-4" />
    if (step.includes('Checking')) return <AlertCircle className="w-4 h-4" />
    if (step.includes('Rewriting')) return <FileText className="w-4 h-4" />
    return <Loader2 className="w-4 h-4" />
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setProcessingSteps([])

    // Send message via WebSocket
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        question: input,
        chat_history: messages.map(msg => ({
          role: msg.role,
          content: msg.content
        }))
      }))
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex gap-6">
        {/* Main Chat Area */}
        <div className="flex-1">
          <div className="card h-[600px] flex flex-col">
            {/* Chat Header */}
            <div className="flex items-center justify-between p-4 border-b border-secondary-200">
              <h2 className="text-xl font-semibold text-secondary-900">
                RAG Chatbot
              </h2>
              <div className="flex items-center space-x-4">
                <label className="flex items-center space-x-2 text-sm">
                  <input
                    type="checkbox"
                    checked={showSteps}
                    onChange={(e) => setShowSteps(e.target.checked)}
                    className="rounded"
                  />
                  <span>Show Steps</span>
                </label>
                <label className="flex items-center space-x-2 text-sm">
                  <input
                    type="checkbox"
                    checked={showSources}
                    onChange={(e) => setShowSources(e.target.checked)}
                    className="rounded"
                  />
                  <span>Show Sources</span>
                </label>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="text-center text-secondary-500 py-8">
                  <Bot className="w-12 h-12 mx-auto mb-4 text-secondary-400" />
                  <p className="text-lg">Start a conversation with your documents</p>
                  <p className="text-sm">Ask questions about the content you've uploaded</p>
                </div>
              )}

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-primary-600 text-white'
                        : 'bg-secondary-100 text-secondary-900'
                    }`}
                  >
                    <div className="flex items-start space-x-2">
                      {message.role === 'user' ? (
                        <User className="w-5 h-5 mt-0.5 flex-shrink-0" />
                      ) : (
                        <Bot className="w-5 h-5 mt-0.5 flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown
                            components={{
                              a: (props) => (
                                <a {...props} className="text-primary-600 underline" />
                              ),
                              code: (props) => (
                                <code {...props} className="bg-secondary-200 px-1 py-0.5 rounded" />
                              )
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                        
                        {/* Metadata */}
                        {message.metadata && (
                          <div className="mt-3 space-y-2">
                            {/* Processing Steps */}
                            {showSteps && message.metadata.steps && (
                              <details className="text-xs">
                                <summary className="cursor-pointer font-medium">
                                  Processing Steps ({message.metadata.steps.length})
                                </summary>
                                <ul className="mt-2 space-y-1">
                                  {message.metadata.steps.map((step, index) => (
                                    <li key={index} className="flex items-center space-x-2">
                                      <CheckCircle className="w-3 h-3 text-green-600" />
                                      <span>{step}</span>
                                    </li>
                                  ))}
                                </ul>
                                <div className="mt-2 text-xs text-secondary-600">
                                  <p>Query rewrites: {message.metadata.rewrites}</p>
                                  <p>Answer grounded: {message.metadata.grounded ? 'Yes' : 'No'}</p>
                                </div>
                              </details>
                            )}

                            {/* Source Documents */}
                            {showSources && message.metadata.documents && message.metadata.documents.length > 0 && (
                              <details className="text-xs">
                                <summary className="cursor-pointer font-medium">
                                  Source Documents ({message.metadata.documents.length})
                                </summary>
                                <div className="mt-2 space-y-2">
                                  {message.metadata.documents.map((doc, index) => (
                                    <div key={index} className="bg-white p-2 rounded border">
                                      <p className="font-medium text-xs mb-1">
                                        Document {index + 1}
                                      </p>
                                      <p className="text-xs text-secondary-600 mb-2">
                                        {doc.content.substring(0, 200)}...
                                      </p>
                                      {doc.metadata && (
                                        <pre className="text-xs text-secondary-500 overflow-x-auto">
                                          {JSON.stringify(doc.metadata, null, 2)}
                                        </pre>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-secondary-100 rounded-lg p-4 max-w-[80%]">
                    <div className="flex items-center space-x-2">
                      <Bot className="w-5 h-5" />
                      <span className="text-secondary-600">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-secondary-200">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about your documents..."
                  className="flex-1 input-field"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isLoading}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Processing Steps Sidebar */}
        {showSteps && processingSteps.length > 0 && (
          <div className="w-80">
            <div className="card h-[600px]">
              <h3 className="text-lg font-semibold text-secondary-900 mb-4">
                Processing Steps
              </h3>
              <div className="space-y-3 overflow-y-auto">
                {processingSteps.map((step) => (
                  <div
                    key={step.id}
                    className={`p-3 rounded-lg border ${
                      step.status === 'active'
                        ? 'border-primary-300 bg-primary-50'
                        : step.status === 'completed'
                        ? 'border-green-300 bg-green-50'
                        : 'border-secondary-200 bg-secondary-50'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      {step.status === 'active' ? (
                        <Loader2 className="w-4 h-4 text-primary-600 animate-spin" />
                      ) : step.status === 'completed' ? (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      ) : (
                        getStepIcon(step.step)
                      )}
                      <span className="text-sm font-medium">{step.step}</span>
                    </div>
                    {step.details && (
                      <div className="mt-2 text-xs text-secondary-600">
                        <pre className="whitespace-pre-wrap">
                          {JSON.stringify(step.details, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}


