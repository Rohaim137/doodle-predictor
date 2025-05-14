'use client';

import { useRef, useState, useEffect, MouseEvent } from 'react';
import * as tf from '@tensorflow/tfjs';

// Class names for the Quick Draw dataset
const CLASS_NAMES = ['cat', 'dog', 'apple', 'banana', 'car', 'house', 'tree', 'bicycle', 'fish', 'chair'];

export default function DoodlePredictor() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Load the model on first render
  useEffect(() => {
    tf.loadGraphModel('/model/model.json').then(setModel);
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Reset last position when clearing
    lastPosRef.current = null;
  };

  const preprocessCanvas = async (canvas: HTMLCanvasElement) => {
    // Step 1: Create a temporary canvas for processing
    const resized = document.createElement('canvas');
    resized.width = 28;
    resized.height = 28;
    const resizedCtx = resized.getContext('2d', { willReadFrequently: true });
    if (!resizedCtx) return null;

    // Step 2: Draw and resize the image
    resizedCtx.fillStyle = 'black';  // Set black background
    resizedCtx.fillRect(0, 0, 28, 28);
    resizedCtx.imageSmoothingEnabled = true;
    resizedCtx.imageSmoothingQuality = 'high';
    resizedCtx.drawImage(canvas, 0, 0, 28, 28);

    // Step 3: Get image data and convert to grayscale
    const imageData = resizedCtx.getImageData(0, 0, 28, 28);
    const { data } = imageData;
    const grayscaleData: number[] = [];

    // Convert to grayscale and normalize to [0,1]
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Convert to grayscale using the same weights as PIL
      const gray = (0.299 * r + 0.587 * g + 0.114 * b);
      // Normalize to [0,1], keeping white strokes on black background
      const normalized = gray / 255.0;
      grayscaleData.push(normalized);
    }

    // Create the tensor with shape [1, 28, 28, 1] for the model
    const input = tf.tensor(grayscaleData, [1, 28, 28, 1]);
    return input;
  };

  const handlePredict = async () => {
    if (!model || !canvasRef.current) return;
    setLoading(true);
    try {
      const inputTensor = await preprocessCanvas(canvasRef.current);
      if (!inputTensor) return;
      const predictions = await model.predict(inputTensor) as tf.Tensor;
      const probs = await predictions.data();
      const index = (await predictions.argMax(-1).data())[0];
      setPrediction(index);
      setProbabilities(Array.from(probs));
      // Cleanup tensors
      inputTensor.dispose();
      predictions.dispose();
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const draw = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (e.buttons !== 1) {
      lastPosRef.current = null;
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (lastPosRef.current) {
      ctx.beginPath();
      ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }

    lastPosRef.current = { x, y };
  };

  const handleMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Draw a dot at the starting point
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();

    lastPosRef.current = { x, y };
  };

  // Initialize black canvas on component mount
  useEffect(() => {
    if (canvasRef.current) {
      clearCanvas();
    }
  }, []);

  return (
    <div className="flex flex-col items-center p-8 gap-6 max-w-4xl mx-auto">
      <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
        Doodle Predictor
      </h1>
      <div className="flex flex-col items-center gap-2">
        <p className="text-gray-400 text-lg">Draw something and let AI guess what it is!</p>
        <div className="flex items-center gap-2 text-emerald-400/90 bg-emerald-400/10 px-4 py-2 rounded-full border border-emerald-400/20">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <span className="text-sm">Keep your doodle simple and centered for better predictions!</span>
        </div>
      </div>
      
      <div className="w-full max-w-2xl bg-gray-800/30 rounded-xl p-6 mb-2 border border-gray-700/50">
        <h2 className="text-lg font-semibold text-blue-400 mb-3">Available Categories</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {CLASS_NAMES.map((category) => (
            <div 
              key={category}
              className="px-3 py-2 bg-gray-700/50 rounded-lg text-center
                       border border-gray-600/50 hover:border-blue-500/50
                       transform hover:scale-105 transition-all duration-200"
            >
              <span className="text-gray-300 font-medium capitalize">{category}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div className="flex flex-col items-center gap-3">
        <label className="text-lg font-semibold text-blue-400">Drawing Canvas</label>
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg blur opacity-40 group-hover:opacity-75 transition duration-200"></div>
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            className="relative border-2 border-gray-700 bg-black cursor-crosshair rounded-lg"
            onMouseMove={draw}
            onMouseDown={handleMouseDown}
          />
        </div>
      </div>

      <div className="flex gap-4 mt-6">
        <button 
          onClick={handlePredict} 
          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg font-semibold 
                   hover:from-blue-600 hover:to-blue-700 transform hover:scale-105 transition-all duration-200 
                   disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 shadow-lg"
          disabled={loading}
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Predicting...
            </span>
          ) : 'Predict Drawing'}
        </button>
        <button 
          onClick={clearCanvas} 
          className="px-6 py-3 bg-gray-700 text-white rounded-lg font-semibold
                   hover:bg-gray-600 transform hover:scale-105 transition-all duration-200
                   focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 shadow-lg"
        >
          Clear Canvas
        </button>
      </div>

      {prediction !== null && probabilities && (
        <div className="flex flex-col items-center gap-4 w-full max-w-2xl mt-4">
          <div className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            I think it&apos;s a <span className="uppercase">{CLASS_NAMES[prediction]}</span>!
          </div>
          <div className="w-full bg-gray-800/50 p-6 rounded-xl shadow-xl border border-gray-700 backdrop-blur-sm">
            <h3 className="text-lg font-semibold text-blue-400 mb-4">Confidence Levels</h3>
            {CLASS_NAMES.map((className, idx) => (
              <div key={className} className="flex justify-between items-center mb-3 group">
                <span className="text-base text-gray-300 font-medium capitalize w-24">{className}</span>
                <div className="flex items-center gap-3 flex-1 ml-4">
                  <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500 ease-out"
                      style={{ width: `${(probabilities[idx] * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm font-mono text-gray-400 w-16">
                    {(probabilities[idx] * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 