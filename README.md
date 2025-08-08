# NeuraViz - Neural Network Visualization & Builder

NeuraViz is a comprehensive web application for building, training, and visualizing neural networks. It features an intuitive drag-and-drop interface for network construction, real-time training visualization, and comprehensive dataset management.

## Features

### Neural Network Builder
- **Intuitive Layer-by-Layer Construction**: Add dense layers, convolutional layers, pooling layers, dropout, and batch normalization
- **Real-time Parameter Configuration**: Adjust layer parameters, activation functions, and hyperparameters
- **Visual Network Architecture**: See your network structure as you build it
- **Support for Multiple Network Types**: Regression, classification, and CNNs

### Dataset Management
- **Built-in Datasets**: Iris, regression samples, classification samples
- **Custom Dataset Upload**: Upload your own CSV datasets
- **Data Visualization**: Preview and explore dataset statistics
- **Automatic Type Detection**: Automatically detects regression vs classification problems

### Training & Visualization
- **Real-time Training Progress**: Live loss and accuracy curves
- **Interactive Charts**: Powered by Recharts for smooth visualizations
- **Training Controls**: Start, stop, and monitor training progress
- **Hyperparameter Configuration**: Adjust learning rate, batch size, optimizer, and loss functions

### Model Management
- **Model Repository**: Save and manage multiple trained models
- **Performance Metrics**: Detailed training history and final metrics
- **Model Export**: Download trained models (future feature)
- **Model Testing**: Evaluate models on test data (future feature)

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Material-UI (MUI)** for modern UI components
- **React Flow** for interactive network diagrams
- **Recharts** for data visualization
- **Emotion** for styling

### Backend
- **FastAPI** for high-performance API
- **PyTorch** for neural network training and inference
- **scikit-learn** for dataset utilities and preprocessing
- **NumPy & Pandas** for data manipulation
- **Uvicorn** as ASGI server

## Getting Started

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NeuraVizNet
   ```

2. **Setup Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Setup Frontend**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   python main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the Frontend Development Server**
   ```bash
   cd frontend
   npm start
   ```
   The web application will open at `http://localhost:3000`

## Usage Guide

### Building Your First Neural Network

1. **Navigate to Network Builder**: Click on the "Network Builder" tab
2. **Add Layers**: Use the layer palette to add layers to your network
3. **Configure Parameters**: Click the settings icon on each layer to adjust parameters
4. **Set Activation Functions**: Choose appropriate activation functions for each layer
5. **Save Network**: Click "Save Network" to store your architecture

### Training a Model

1. **Go to Training & Visualization**: Click on the "Training & Visualization" tab
2. **Select Model and Dataset**: Choose from your saved models and available datasets
3. **Configure Training**: Set epochs, batch size, learning rate, and optimizer
4. **Start Training**: Click "Start Training" to begin the process
5. **Monitor Progress**: Watch real-time loss and accuracy curves

### Managing Datasets

1. **Access Dataset Manager**: Click on the "Datasets" tab
2. **Browse Built-in Datasets**: Explore pre-loaded datasets like Iris
3. **Upload Custom Data**: Click "Upload Dataset" to add your own CSV files
4. **Preview Data**: Click "View Details" to examine dataset structure and sample data

## Supported Layer Types

| Layer Type | Description | Parameters |
|------------|-------------|------------|
| **Dense/Linear** | Fully connected layer | Units, activation function |
| **Convolutional 2D** | 2D convolution for image processing | Filters, kernel size, padding |
| **Max Pooling 2D** | Downsampling layer | Pool size |
| **Dropout** | Regularization layer | Dropout rate |
| **Batch Normalization** | Normalization layer | - |
| **Flatten** | Reshape layer for CNN to dense transition | - |

## Supported Activation Functions

- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Sigmoid
- Leaky ReLU
- ELU (Exponential Linear Unit)
- Swish
- GELU (Gaussian Error Linear Unit)

## API Documentation

When the backend is running, visit `http://localhost:8000/docs` for interactive API documentation powered by FastAPI's automatic OpenAPI generation.

## Development

### Project Structure
```
NeuraVizNet/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── App.tsx         # Main application
│   │   └── index.tsx       # Entry point
├── backend/                 # FastAPI backend
│   ├── main.py             # FastAPI application
│   └── requirements.txt    # Python dependencies
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] **Advanced Visualizations**: Layer activation maps, weight visualizations
- [ ] **Model Comparison**: Side-by-side model performance comparison
- [ ] **Hyperparameter Optimization**: Automated hyperparameter tuning
- [ ] **Model Export**: Export to ONNX, TensorFlow formats
- [ ] **Collaborative Features**: Share models and datasets with teams
- [ ] **Advanced CNN Architectures**: ResNet, VGG, custom architectures
- [ ] **Transfer Learning**: Pre-trained model integration
- [ ] **Real-time Inference**: Live prediction interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern web technologies for optimal performance
- Inspired by the need for intuitive neural network design tools
- Community-driven development for educational and research purposes

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository or contact the development team.

---

**Happy Neural Networking!** 
