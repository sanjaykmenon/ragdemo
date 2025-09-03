'use client';

import { useChat } from 'ai/react';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Upload, FileText, Loader2 } from 'lucide-react';

interface Document {
  id: number;
  title: string;
  content: string;
  similarity?: number;
}

export default function Home() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    
    // Read file content
    const text = await file.text();
    
    // Send a message to add the document
    const addDocumentMessage = `Add this financial document to the knowledge base:
Title: ${file.name}
Content: ${text}`;
    
    handleSubmit({
      preventDefault: () => {},
      target: { value: addDocumentMessage }
    } as any);
    
    setSelectedFile(null);
    event.target.value = '';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Financial Documents RAG Demo
          </h1>
          <p className="text-lg text-gray-600">
            Upload financial documents and ask questions about them using AI-powered retrieval
          </p>
        </motion.div>

        <div className="bg-white rounded-lg shadow-lg border border-gray-200">
          {/* File Upload Section */}
          <div className="border-b border-gray-200 p-4">
            <label className="flex items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors">
              <div className="flex flex-col items-center">
                <Upload className="w-8 h-8 text-gray-400 mb-2" />
                <span className="text-sm text-gray-600">
                  Click to upload financial documents (PDF, TXT, etc.)
                </span>
              </div>
              <input
                type="file"
                className="hidden"
                onChange={handleFileUpload}
                accept=".pdf,.txt,.docx,.csv"
              />
            </label>
            {selectedFile && (
              <div className="mt-2 text-sm text-blue-600">
                Uploading: {selectedFile.name}
              </div>
            )}
          </div>

          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-20">
                <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>Start by uploading some financial documents or asking a question!</p>
              </div>
            )}
            
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{message.content}</div>
                  
                  {/* Display tool results */}
                  {message.toolInvocations?.map((toolInvocation) => (
                    <div key={toolInvocation.toolCallId} className="mt-2 text-sm">
                      {toolInvocation.toolName === 'retrieveInformation' && toolInvocation.result && (
                        <div className="bg-blue-50 p-2 rounded border">
                          <strong>Retrieved Documents:</strong>
                          {toolInvocation.result.documents?.map((doc: Document) => (
                            <div key={doc.id} className="mt-1 p-2 bg-white rounded border">
                              <div className="font-medium">{doc.title}</div>
                              <div className="text-xs text-gray-600">
                                Similarity: {(doc.similarity * 100).toFixed(1)}%
                              </div>
                              <div className="text-sm mt-1">{doc.content}</div>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {toolInvocation.toolName === 'addDocument' && toolInvocation.result && (
                        <div className="bg-green-50 p-2 rounded border">
                          {toolInvocation.result.success ? (
                            <span className="text-green-700">
                              ✓ Document added successfully: {toolInvocation.result.document?.title}
                            </span>
                          ) : (
                            <span className="text-red-700">
                              ✗ Failed to add document: {toolInvocation.result.error}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-gray-100 text-gray-900 px-4 py-2 rounded-lg flex items-center">
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Thinking...
                </div>
              </motion.div>
            )}
          </div>

          {/* Chat Input */}
          <form onSubmit={handleSubmit} className="border-t border-gray-200 p-4">
            <div className="flex space-x-2">
              <input
                value={input}
                onChange={handleInputChange}
                placeholder="Ask about your financial documents..."
                className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>

        {/* Sample Questions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mt-8 bg-white rounded-lg shadow-lg border border-gray-200 p-6"
        >
          <h3 className="text-lg font-semibold mb-4">Sample Questions to Try:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              "What are the key financial metrics in the latest quarterly report?",
              "Summarize the revenue trends from the uploaded documents",
              "What are the main risk factors mentioned?",
              "Show me information about cash flow and liquidity",
            ].map((question, index) => (
              <button
                key={index}
                onClick={() => handleInputChange({ target: { value: question } } as any)}
                className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg text-sm text-gray-700 transition-colors"
              >
                {question}
              </button>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
