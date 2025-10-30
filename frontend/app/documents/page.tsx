'use client'

import { useState, useEffect } from 'react'
import { FileText, Calendar, Download, Trash2 } from 'lucide-react'
import axios from 'axios'

interface Document {
  id: string
  name: string
  size: number
  uploadDate: string
  status: 'processed' | 'processing' | 'error'
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Simulate loading documents
    setTimeout(() => {
      setDocuments([
        {
          id: '1',
          name: 'research_paper.pdf',
          size: 2048576,
          uploadDate: '2024-01-15T10:30:00Z',
          status: 'processed'
        },
        {
          id: '2',
          name: 'user_manual.docx',
          size: 1536000,
          uploadDate: '2024-01-14T15:45:00Z',
          status: 'processed'
        },
        {
          id: '3',
          name: 'technical_specs.txt',
          size: 512000,
          uploadDate: '2024-01-13T09:20:00Z',
          status: 'processed'
        }
      ])
      setIsLoading(false)
    }, 1000)
  }, [])

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processed':
        return 'text-green-600 bg-green-100'
      case 'processing':
        return 'text-yellow-600 bg-yellow-100'
      case 'error':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-secondary-600 bg-secondary-100'
    }
  }

  const deleteDocument = async (id: string) => {
    if (confirm('Are you sure you want to delete this document?')) {
      setDocuments(prev => prev.filter(doc => doc.id !== id))
    }
  }

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-secondary-600">Loading documents...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900 mb-2">
            Documents
          </h1>
          <p className="text-secondary-600">
            Manage your uploaded documents and view processing status
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-secondary-500">
            Total Documents: {documents.length}
          </p>
          <p className="text-sm text-secondary-500">
            Processed: {documents.filter(doc => doc.status === 'processed').length}
          </p>
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="card text-center py-12">
          <FileText className="w-16 h-16 mx-auto mb-4 text-secondary-400" />
          <h3 className="text-xl font-semibold text-secondary-900 mb-2">
            No Documents Uploaded
          </h3>
          <p className="text-secondary-600 mb-6">
            Upload some documents to start building your knowledge base
          </p>
          <a href="/" className="btn-primary">
            Upload Documents
          </a>
        </div>
      ) : (
        <div className="space-y-4">
          {documents.map((doc) => (
            <div key={doc.id} className="card">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                    <FileText className="w-5 h-5 text-primary-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-secondary-900">
                      {doc.name}
                    </h3>
                    <div className="flex items-center space-x-4 text-sm text-secondary-500">
                      <span>{formatFileSize(doc.size)}</span>
                      <span className="flex items-center space-x-1">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(doc.uploadDate)}</span>
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(doc.status)}`}>
                    {doc.status}
                  </span>
                  
                  <div className="flex space-x-2">
                    <button className="p-2 text-secondary-400 hover:text-secondary-600 transition-colors">
                      <Download className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={() => deleteDocument(doc.id)}
                      className="p-2 text-secondary-400 hover:text-red-500 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}




