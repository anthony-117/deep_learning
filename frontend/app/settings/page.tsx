'use client'

import { useState, useEffect } from 'react'
import { Settings, Database, Brain, Key, Save, CheckCircle, AlertCircle } from 'lucide-react'
import axios from 'axios'

interface SystemStatus {
  rag_ready: boolean
  vector_store_ready: boolean
  embedding_ready: boolean
  llm_ready: boolean
}

interface Config {
  groq_api_key: string
  milvus_host: string
  milvus_port: string
  vector_db_collection: string
  embedding_model: string
  generation_model: string
  temperature: number
  top_k: number
}

export default function SettingsPage() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [config, setConfig] = useState<Config>({
    groq_api_key: '',
    milvus_host: 'localhost',
    milvus_port: '19530',
    vector_db_collection: 'docling_demo',
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    generation_model: 'meta-llama/llama-4-scout-17b-16e-instruct',
    temperature: 0.05,
    top_k: 3
  })
  const [isSaving, setIsSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'success' | 'error'>('idle')

  useEffect(() => {
    // Load system status
    axios.get('http://localhost:8000/health')
      .then(response => {
        setSystemStatus(response.data)
      })
      .catch(error => {
        console.error('Error loading system status:', error)
      })

    // Load current config (this would typically come from your backend)
    // For now, we'll use the default values
  }, [])

  const handleSave = async () => {
    setIsSaving(true)
    setSaveStatus('idle')

    try {
      // This would typically save to your backend
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      
      setSaveStatus('success')
      setTimeout(() => setSaveStatus('idle'), 3000)
    } catch (error) {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus('idle'), 3000)
    } finally {
      setIsSaving(false)
    }
  }

  const getStatusIcon = (status: boolean) => {
    return status ? (
      <CheckCircle className="w-5 h-5 text-green-600" />
    ) : (
      <AlertCircle className="w-5 h-5 text-red-600" />
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-secondary-900 mb-2">
          Settings
        </h1>
        <p className="text-secondary-600">
          Configure your RAG system and view system status
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* System Status */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-6">
            <Settings className="w-5 h-5 text-primary-600" />
            <h2 className="text-xl font-semibold text-secondary-900">
              System Status
            </h2>
          </div>

          {systemStatus ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Brain className="w-4 h-4 text-secondary-600" />
                  <span className="font-medium">RAG System</span>
                </div>
                {getStatusIcon(systemStatus.rag_ready)}
              </div>

              <div className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Database className="w-4 h-4 text-secondary-600" />
                  <span className="font-medium">Vector Store</span>
                </div>
                {getStatusIcon(systemStatus.vector_store_ready)}
              </div>

              <div className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Brain className="w-4 h-4 text-secondary-600" />
                  <span className="font-medium">Embedding Model</span>
                </div>
                {getStatusIcon(systemStatus.embedding_ready)}
              </div>

              <div className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Key className="w-4 h-4 text-secondary-600" />
                  <span className="font-medium">LLM Model</span>
                </div>
                {getStatusIcon(systemStatus.llm_ready)}
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-secondary-600">Loading system status...</p>
            </div>
          )}
        </div>

        {/* Configuration */}
        <div className="card">
          <div className="flex items-center space-x-2 mb-6">
            <Settings className="w-5 h-5 text-primary-600" />
            <h2 className="text-xl font-semibold text-secondary-900">
              Configuration
            </h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                GROQ API Key
              </label>
              <input
                type="password"
                value={config.groq_api_key}
                onChange={(e) => setConfig(prev => ({ ...prev, groq_api_key: e.target.value }))}
                className="input-field"
                placeholder="Enter your GROQ API key"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Milvus Host
                </label>
                <input
                  type="text"
                  value={config.milvus_host}
                  onChange={(e) => setConfig(prev => ({ ...prev, milvus_host: e.target.value }))}
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Milvus Port
                </label>
                <input
                  type="text"
                  value={config.milvus_port}
                  onChange={(e) => setConfig(prev => ({ ...prev, milvus_port: e.target.value }))}
                  className="input-field"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Vector DB Collection
              </label>
              <input
                type="text"
                value={config.vector_db_collection}
                onChange={(e) => setConfig(prev => ({ ...prev, vector_db_collection: e.target.value }))}
                className="input-field"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Embedding Model
              </label>
              <input
                type="text"
                value={config.embedding_model}
                onChange={(e) => setConfig(prev => ({ ...prev, embedding_model: e.target.value }))}
                className="input-field"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">
                Generation Model
              </label>
              <input
                type="text"
                value={config.generation_model}
                onChange={(e) => setConfig(prev => ({ ...prev, generation_model: e.target.value }))}
                className="input-field"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Temperature
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="2"
                  value={config.temperature}
                  onChange={(e) => setConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-secondary-700 mb-2">
                  Top K
                </label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.top_k}
                  onChange={(e) => setConfig(prev => ({ ...prev, top_k: parseInt(e.target.value) }))}
                  className="input-field"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-4">
              <div className="flex items-center space-x-2">
                {saveStatus === 'success' && (
                  <div className="flex items-center space-x-2 text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">Settings saved!</span>
                  </div>
                )}
                {saveStatus === 'error' && (
                  <div className="flex items-center space-x-2 text-red-600">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm">Failed to save settings</span>
                  </div>
                )}
              </div>
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSaving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Settings
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}




