import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { Upload, Camera, CheckCircle, XCircle, Loader, Image as ImageIcon } from 'lucide-react';
import './App.css';

// API URL from environment variable or fallback to localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      setError(null);
      setResult(null);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const handleSubmit = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Image = reader.result;

        try {
          const response = await axios.post(`${API_URL}/api/predict`, {
            image: base64Image
          });

          if (response.data.success) {
            setResult(response.data);
          } else {
            setError(response.data.error || 'Failed to recognize license plate');
          }
        } catch (err) {
          setError(err.response?.data?.error || 'Error connecting to server. Make sure the backend is running.');
        } finally {
          setLoading(false);
        }
      };
      reader.readAsDataURL(selectedImage);
    } catch (err) {
      setError('Error processing image');
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Camera className="h-8 w-8 text-indigo-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                License Plate Recognition
              </h1>
            </div>
            <div className="text-sm text-gray-500">
              AI-Powered CNN Model
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-4 text-gray-800">
                Upload Image
              </h2>
              
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={`border-3 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                  isDragActive
                    ? 'border-indigo-500 bg-indigo-50'
                    : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-16 w-16 text-gray-400 mb-4" />
                {isDragActive ? (
                  <p className="text-lg text-indigo-600">Drop the image here...</p>
                ) : (
                  <div>
                    <p className="text-lg text-gray-700 mb-2">
                      Drag & drop an image here
                    </p>
                    <p className="text-sm text-gray-500">
                      or click to select a file
                    </p>
                    <p className="text-xs text-gray-400 mt-2">
                      Supports: JPG, JPEG, PNG
                    </p>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="mt-6 flex space-x-4">
                <button
                  onClick={handleSubmit}
                  disabled={!selectedImage || loading}
                  className={`flex-1 flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all ${
                    !selectedImage || loading
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-md hover:shadow-lg'
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader className="animate-spin h-5 w-5" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="h-5 w-5" />
                      <span>Recognize Plate</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={handleClear}
                  disabled={!selectedImage && !result}
                  className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                    !selectedImage && !result
                      ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Clear
                </button>
              </div>
            </div>

            {/* Instructions */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-lg font-semibold mb-3 text-gray-800">
                How to Use
              </h3>
              <ol className="space-y-2 text-sm text-gray-600">
                <li className="flex items-start">
                  <span className="font-bold text-indigo-600 mr-2">1.</span>
                  <span>Upload an image containing a vehicle with a visible license plate</span>
                </li>
                <li className="flex items-start">
                  <span className="font-bold text-indigo-600 mr-2">2.</span>
                  <span>Click "Recognize Plate" to process the image</span>
                </li>
                <li className="flex items-start">
                  <span className="font-bold text-indigo-600 mr-2">3.</span>
                  <span>View the detected plate number and confidence score</span>
                </li>
              </ol>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Image Preview */}
            {imagePreview && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-2xl font-semibold mb-4 text-gray-800 flex items-center">
                  <ImageIcon className="mr-2 h-6 w-6 text-indigo-600" />
                  Image Preview
                </h2>
                <div className="rounded-lg overflow-hidden border-2 border-gray-200">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="w-full h-auto"
                  />
                </div>
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-2xl font-semibold mb-4 text-gray-800 flex items-center">
                  <CheckCircle className="mr-2 h-6 w-6 text-green-600" />
                  Detection Result
                </h2>
                
                <div className="space-y-4">
                  <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-center">
                    <p className="text-sm text-indigo-100 mb-2">License Plate Number</p>
                    <p className="text-4xl font-bold text-white tracking-wider">
                      {result.plate_number}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
                      <div className="flex items-center">
                        <div className="flex-1 bg-gray-200 rounded-full h-3 mr-3">
                          <div
                            className="bg-green-500 h-3 rounded-full transition-all"
                            style={{ width: `${result.confidence * 100}%` }}
                          ></div>
                        </div>
                        <p className="text-lg font-bold text-gray-800">
                          {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {result.bounding_box && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-600 mb-1">Bounding Box</p>
                        <p className="text-sm font-mono text-gray-800">
                          x: {result.bounding_box[0]}, y: {result.bounding_box[1]}, 
                          w: {result.bounding_box[2]}, h: {result.bounding_box[3]}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center text-red-600 mb-2">
                  <XCircle className="mr-2 h-6 w-6" />
                  <h3 className="text-xl font-semibold">Error</h3>
                </div>
                <p className="text-gray-700">{error}</p>
                <p className="text-sm text-gray-500 mt-2">
                  API URL: {API_URL}
                </p>
              </div>
            )}

            {/* Placeholder */}
            {!imagePreview && !result && !error && (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <Camera className="mx-auto h-20 w-20 text-gray-300 mb-4" />
                <p className="text-gray-500 text-lg">
                  Upload an image to get started
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white mt-12 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-gray-500 text-sm">
            Built with React, TensorFlow, and Flask • CNN-based License Plate Recognition System
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
