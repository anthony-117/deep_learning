'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import axios from 'axios'

interface UploadProgress {
  status: 'idle' | 'uploading' | 'processing' | 'success' | 'error'
  message: string
  progress: number
}

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    status: 'idle',
    message: '',
    progress: 0
  })

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/markdown': ['.md'],
    },
    multiple: true
  })

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (files.length === 0) return

    setUploadProgress({
      status: 'uploading',
      message: 'Uploading files...',
      progress: 0
    })

    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 1)
          )
          setUploadProgress(prev => ({
            ...prev,
            progress,
            message: `Uploading... ${progress}%`
          }))
        }
      })

      setUploadProgress({
        status: 'processing',
        message: 'Processing documents...',
        progress: 100
      })

      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000))

      setUploadProgress({
        status: 'success',
        message: `Successfully processed ${response.data.files_processed} files!`,
        progress: 100
      })

      // Clear files after successful upload
      setFiles([])

    } catch (error: any) {
      setUploadProgress({
        status: 'error',
        message: error.response?.data?.detail || 'Upload failed',
        progress: 0
      })
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-secondary-900 mb-4">
          Upload Documents
        </h1>
        <p className="text-secondary-600 text-lg">
          Upload your documents to build a knowledge base for the RAG chatbot
        </p>
      </div>

      {/* Upload Area */}
      <div className="card mb-8">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors duration-200 ${
            isDragActive
              ? 'border-primary-400 bg-primary-50'
              : 'border-secondary-300 hover:border-primary-400 hover:bg-primary-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 mx-auto mb-4 text-secondary-400" />
          <h3 className="text-xl font-semibold text-secondary-900 mb-2">
            {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
          </h3>
          <p className="text-secondary-600 mb-4">
            or click to select files
          </p>
          <p className="text-sm text-secondary-500">
            Supports PDF, DOC, DOCX, TXT, and MD files
          </p>
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="card mb-8">
          <h3 className="text-lg font-semibold text-secondary-900 mb-4">
            Selected Files ({files.length})
          </h3>
          <div className="space-y-3">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-secondary-50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <FileText className="w-5 h-5 text-secondary-600" />
                  <div>
                    <p className="font-medium text-secondary-900">{file.name}</p>
                    <p className="text-sm text-secondary-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="text-secondary-400 hover:text-red-500 transition-colors"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Progress */}
      {uploadProgress.status !== 'idle' && (
        <div className="card mb-8">
          <div className="flex items-center space-x-3 mb-4">
            {uploadProgress.status === 'uploading' && (
              <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
            )}
            {uploadProgress.status === 'processing' && (
              <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />
            )}
            {uploadProgress.status === 'success' && (
              <CheckCircle className="w-5 h-5 text-green-600" />
            )}
            {uploadProgress.status === 'error' && (
              <AlertCircle className="w-5 h-5 text-red-600" />
            )}
            <span className="font-medium text-secondary-900">
              {uploadProgress.message}
            </span>
          </div>
          
          {uploadProgress.status === 'uploading' && (
            <div className="w-full bg-secondary-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress.progress}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Upload Button */}
      {files.length > 0 && uploadProgress.status === 'idle' && (
        <div className="text-center">
          <button
            onClick={uploadFiles}
            className="btn-primary text-lg px-8 py-3"
          >
            Upload & Process Documents
          </button>
        </div>
      )}

      {/* Success Message */}
      {uploadProgress.status === 'success' && (
        <div className="text-center">
          <div className="card bg-green-50 border-green-200">
            <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-green-900 mb-2">
              Upload Successful!
            </h3>
            <p className="text-green-700 mb-4">
              Your documents have been processed and are ready for chat.
            </p>
            <a
              href="/chat"
              className="btn-primary"
            >
              Start Chatting
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
