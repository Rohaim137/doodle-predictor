# Doodle Predictor

A fun, interactive web application that uses machine learning to predict what you're drawing in real-time. Built with Next.js, TensorFlow.js, and Tailwind CSS.
Deployed at: https://doodle-predictor.vercel.app/
## Features

- 🎨 Real-time drawing canvas
- 🤖 ML-powered predictions
- 🎯 Supports 10 different categories:
  - Cat
  - Dog
  - Apple
  - Banana
  - Car
  - House
  - Tree
  - Bicycle
  - Fish
  - Chair
- 💫 Beautiful, modern UI with smooth animations
- 📱 Responsive design that works on all devices

## Tech Stack

- **Frontend Framework**: [Next.js](https://nextjs.org/)
- **Machine Learning**: [TensorFlow.js](https://www.tensorflow.org/js)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Language**: TypeScript

## Getting Started

### Prerequisites

- Node.js 18.0 or later
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Rohaim137/doodle-predictor.git
cd doodle-predictor
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage

1. Open the application in your web browser
2. Choose one of the available categories to draw
3. Draw your doodle in the canvas area
   - Keep it simple and centered for better predictions
   - Use clear, distinct lines
4. Click the "Predict" button to see what the AI thinks you've drawn
5. Use the "Clear" button to start over

## Model Information

The prediction model is trained on the [Quick Draw dataset](https://github.com/googlecreativelab/quickdraw-dataset) and converted to TensorFlow.js format. It uses a convolutional neural network (CNN) architecture to recognize hand-drawn sketches.


## Acknowledgments

- [Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset) for the training data
- [TensorFlow.js](https://www.tensorflow.org/js) for the machine learning capabilities
- [Next.js](https://nextjs.org/) for the web framework
- [Vercel](https://vercel.com) for hosting
