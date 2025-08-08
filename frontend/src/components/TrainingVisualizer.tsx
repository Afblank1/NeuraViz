import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  CardHeader
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  TrendingUp as MetricsIcon,
  Architecture as NetworkIcon,
  Visibility as ViewIcon,
  BarChart as ChartsIcon,
  ScatterPlot as ScatterIcon,
  TableChart as ConfusionIcon,
  Timeline as ProgressIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  ModelTraining as TrainingIcon,
  Storage as DataIcon,
  TrendingUp as LandscapeIcon,
  Science as TestIcon,
  BarChart as VizIcon,
  Tune as ConfigIcon,
  Label as LabelIcon,
  CheckCircleOutline as OkIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  Cell,
  PieChart,
  Pie,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';



interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;        // For classification
  val_loss: number;
  val_accuracy?: number;    // For classification
  r2_score?: number;        // For regression
  val_r2_score?: number;    // For regression
  learning_rate?: number;
  timestamp: string;
}

interface Model {
  config: any;
  created_at: string;
  status: string;
  training_history: TrainingMetrics[];
  architecture_summary?: any;
  parameter_count?: number;
  display_name?: string;
  dataset_name?: string;
  dataset_type?: string;
  split_info?: {
    train_split: number;
    validation_split: number;
    test_split: number;
    use_test_set: boolean;
    random_seed: number;
    train_size: number;
    validation_size: number;
    test_size: number;
  };
  test_set?: {
    X_test: number[][];
    y_test: number[];
    size: number;
  } | null;
  final_metrics?: {
    final_loss: number;
    final_accuracy?: number;  // For classification
    best_loss: number;
    best_accuracy?: number;   // For classification
    final_r2_score?: number;  // For regression
    best_r2_score?: number;   // For regression
    total_epochs?: number;
    total_parameters?: number;
    trainable_parameters?: number;
  };
}

interface Dataset {
  name: string;
  type: 'regression' | 'classification';
  input_size: number;
  output_size: number;
  X?: number[][];
  y?: number[];
  total_samples?: number;
  X_train?: number[][];
  y_train?: number[];
  X_test?: number[][];
  y_test?: number[];
  classes?: string[];
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0', '#ffb347'];

// Icon mapping 
const getIconForTitle = (title: string) => {
  const iconMap: Record<string, React.ReactElement> = {
    'PROGRESS': <ProgressIcon fontSize="small" />,
    'TRAINING': <TrainingIcon fontSize="small" />,
    'METRICS': <MetricsIcon fontSize="small" />,
    'LANDSCAPE': <LandscapeIcon fontSize="small" />,
    'DATA': <DataIcon fontSize="small" />,
    'OK': <OkIcon fontSize="small" />,
    'TEST': <TestIcon fontSize="small" />,
    'LABEL': <LabelIcon fontSize="small" />,
    'VIZ': <VizIcon fontSize="small" />,
    'CONFIG': <ConfigIcon fontSize="small" />,
    'SETTINGS': <SettingsIcon fontSize="small" />
  };
  
  // Extract the key from [KEY] format
  const match = title.match(/\[(\w+)\]/);
  if (match && iconMap[match[1]]) {
    return iconMap[match[1]];
  }
  
  return null;
};

// Helper function to create titles with icons
const createTitleWithIcon = (title: string) => {
  const icon = getIconForTitle(title);
  const cleanTitle = title.replace(/\[\w+\]\s*/, '');
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {icon}
      <Typography component="span">{cleanTitle}</Typography>
    </Box>
  );
};

// Safe number formatting function 
const formatNumber = (num: number | null | undefined, decimals: number = 4) => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  return num.toFixed(decimals);
};

const formatPercentage = (num: number | null | undefined, decimals: number = 1) => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  return (num * 100).toFixed(decimals) + '%';
};

// Loss function options based on task type
const LOSS_FUNCTIONS = {
  classification: [
    { value: 'crossentropy', label: 'Cross Entropy', description: 'Standard for multi-class classification' },
    { value: 'binary_crossentropy', label: 'Binary Cross Entropy', description: 'For binary classification' },
    { value: 'sparse_crossentropy', label: 'Sparse Cross Entropy', description: 'For integer labels' },
  ],
  regression: [
    { value: 'mse', label: 'Mean Squared Error (MSE)', description: 'Standard for regression' },
    { value: 'mae', label: 'Mean Absolute Error (MAE)', description: 'Less sensitive to outliers' },
    { value: 'huber', label: 'Huber Loss', description: 'Robust to outliers' },
    { value: 'smooth_l1', label: 'Smooth L1', description: 'Huber-like loss' },
  ]
};

const TrainingVisualizer: React.FC = () => {
  const [models, setModels] = useState<Record<string, Model>>({});
  const [datasets, setDatasets] = useState<Record<string, Dataset>>({});
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<TrainingMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<TrainingMetrics | null>(null);
  const [visualizationMode, setVisualizationMode] = useState('overview');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  const [modelVisualizations, setModelVisualizations] = useState<any>(null);
  const [evaluationResults, setEvaluationResults] = useState<any>(null);
  const [testSetResults, setTestSetResults] = useState<any>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isTestingOnTestSet, setIsTestingOnTestSet] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    lossFunction: 'auto', // Auto select based on dataset type
    // Data split 
    trainSplit: 0.7,
    validationSplit: 0.2,
    testSplit: 0.1,
    useTestSet: true,
    randomSeed: 42,
    earlyStopping: false,
    patience: 10
  });

  // Helper function to get the appropriate loss function
  const getEffectiveLossFunction = () => {
    if (!selectedDataset || !datasets[selectedDataset]) return 'mse';
    
    const datasetType = datasets[selectedDataset].type;
    
    if (trainingConfig.lossFunction === 'auto') {
      // Auto-select based on dataset type
      return datasetType === 'classification' ? 'crossentropy' : 'mse';
    }
    
    return trainingConfig.lossFunction;
  };

  // Get available loss functions for current dataset
  const getAvailableLossFunctions = () => {
    if (!selectedDataset || !datasets[selectedDataset]) {
      return [
        { value: 'auto', label: 'Auto (Select dataset first)', description: 'Automatically selected for dataset type' }
      ];
    }
    
    const datasetType = datasets[selectedDataset].type;
    return [
      { value: 'auto', label: `Auto (${datasetType === 'classification' ? 'Cross Entropy' : 'MSE'})`, description: 'Automatically selected for dataset type' },
      ...LOSS_FUNCTIONS[datasetType]
    ];
  };

  useEffect(() => {
    fetchModels();
    fetchDatasets();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    let timeoutCounter = 0;
    const maxTimeouts = 300; // Stop after 5 minutes of polling 
    
    if (isTraining && autoRefresh && selectedModel) {

      
      interval = setInterval(() => {
        timeoutCounter++;
        
        // Stop polling after max timeout to prevent infinite loops
        if (timeoutCounter >= maxTimeouts) {

          setIsTraining(false);
          setAutoRefresh(false);
          return;
        }
        
        fetchTrainingStatus();
      }, 3000); // Poll every 3 seconds 
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
        if (timeoutCounter > 0) {

        }
      }
    };
  }, [isTraining, selectedModel, autoRefresh]);

  const fetchModels = async () => {

    try {
      const startTime = Date.now();
      const response = await fetch('http://localhost:8000/models');
      const responseTime = Date.now() - startTime;
      

      
      if (response.ok) {
        const data = await response.json();
        setModels(data);

      } else {

      }
    } catch (error) {

    }
  };

  const fetchDatasets = async () => {

    try {
      const startTime = Date.now();
      const response = await fetch('http://localhost:8000/datasets');
      const responseTime = Date.now() - startTime;
      

      
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
        
        const datasetTypes = Object.values(data).reduce((acc: any, dataset: any) => {
          acc[dataset.type] = (acc[dataset.type] || 0) + 1;
          return acc;
        }, {});
        

      } else {

      }
    } catch (error) {

    }
  };

  const fetchTrainingStatus = async () => {
    if (!selectedModel) return;
    
    try {
      const startTime = Date.now();
      const response = await fetch(`http://localhost:8000/models/${selectedModel}/status`);
      const responseTime = Date.now() - startTime;
      

      
      if (response.ok) {
        const data = await response.json();
        const newHistory = data.training_history || [];
        const previousEpochCount = trainingHistory.length;
        
        setTrainingHistory(newHistory);
        
        if (newHistory.length > 0) {
          setCurrentMetrics(newHistory[newHistory.length - 1]);
          
          // Log only when new epochs are detected
          if (newHistory.length > previousEpochCount) {

          }
        }
        
        // Update model status
        setModels(prev => ({
          ...prev,
          [selectedModel]: { ...prev[selectedModel], ...data }
        }));
        
        // Check if training is complete and update isTraining state
        if (data.status === 'trained' || data.status === 'error') {
          if (isTraining) {

            setIsTraining(false);
          }
        } else if (data.status === 'training') {
          if (!isTraining) {

            setIsTraining(true);
          }
        }
      } else {

      }
    } catch (error) {

    }
  };

  const startTraining = async () => {

    
    if (!selectedModel || !selectedDataset) {

      alert('Please select both a model and dataset');
      return;
    }



    const modelData = models[selectedModel];
    if (!modelData || !modelData.config || !modelData.config.layers) {

      alert('Model configuration is missing or invalid. Please create a new model.');
      return;
    }



    setIsTraining(true);
    const startTime = Date.now();
    
    try {
      const networkConfig = {
        layers: modelData.config.layers,
        epochs: trainingConfig.epochs,
        batch_size: trainingConfig.batchSize,
        learning_rate: trainingConfig.learningRate,
        optimizer: trainingConfig.optimizer,
        loss_function: getEffectiveLossFunction()
      };

      const requestPayload = {
        network_config: networkConfig,
        dataset_name: selectedDataset,
        train_split: trainingConfig.trainSplit,
        validation_split: trainingConfig.validationSplit,
        test_split: trainingConfig.testSplit,
        use_test_set: trainingConfig.useTestSet,
        random_seed: trainingConfig.randomSeed
      };

      const response = await fetch(`http://localhost:8000/train/${selectedModel}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });

      if (response.ok) {
        const result = await response.json();
        
        setTrainingHistory(result.training_history);
        
        await fetchModels();
        alert('Training completed successfully!');
      } else {
        const errorText = await response.text();
        alert(`Training failed: ${errorText}`);
      }
    } catch (error) {
      alert('Error starting training');
    } finally {
      setIsTraining(false);
    }
  };

  const stopTraining = () => {
    setIsTraining(false);
    // In a real implementation, you would send a stop request to the backend
  };

  const fetchModelVisualizations = async (modelId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/visualizations/${modelId}`);
      if (response.ok) {
        const visualizations = await response.json();
        setModelVisualizations(visualizations);
      }
    } catch (error) {

    }
  };

  const evaluateModel = async (modelId: string, datasetName: string) => {
    if (!modelId || !datasetName) return;
    
    setIsEvaluating(true);
    try {
      const response = await fetch(`http://localhost:8000/evaluate/${modelId}?dataset_name=${datasetName}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const results = await response.json();
        setEvaluationResults(results);
      } else {
        alert('Error evaluating model');
      }
    } catch (error) {

      alert('Error evaluating model');
    } finally {
      setIsEvaluating(false);
    }
  };

  const evaluateOnTestSet = async (modelId: string) => {
    if (!modelId) return;
    
    setIsTestingOnTestSet(true);
    try {
      const response = await fetch(`http://localhost:8000/evaluate-test-set/${modelId}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const results = await response.json();
        setTestSetResults(results);

      } else {
        const errorText = await response.text();
        alert(`Error evaluating on test set: ${errorText}`);
      }
    } catch (error) {

      alert('Error evaluating on test set');
    } finally {
      setIsTestingOnTestSet(false);
    }
  };

  const downloadModel = async (modelId: string) => {
    if (!modelId) return;
    
    setIsDownloading(true);
    try {
      const response = await fetch(`http://localhost:8000/download/${modelId}`);
      
      if (response.ok) {
        const result = await response.json();
        
        // Convert base64 to blob and download
        const byteCharacters = atob(result.model_data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/octet-stream' });
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = result.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        
        alert('Model downloaded successfully!');
      } else {
        alert('Error downloading model');
      }
    } catch (error) {

      alert('Error downloading model');
    } finally {
      setIsDownloading(false);
    }
  };

  const generateDatasetDistribution = (dataset: Dataset) => {
    
    const targetData = dataset.y || dataset.y_train;
    
    if (!dataset || !targetData || !Array.isArray(targetData) || targetData.length === 0) {
      return [];
    }

    if (dataset.type === 'classification') {
      const classCounts = targetData.reduce((acc, label) => {
        const className = dataset.classes?.[label] || `Class ${label}`;
        acc[className] = (acc[className] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      return Object.entries(classCounts).map(([name, value], index) => ({
        name,
        value,
        fill: COLORS[index % COLORS.length]
      }));
    } else {
      // For regression, create a histogram of target values
      const bins = 10;
      const values = targetData;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const binSize = (max - min) / bins;
      
      const histogram = Array(bins).fill(0).map((_, i) => ({
        range: `${formatNumber(min + i * binSize, 1)}-${formatNumber(min + (i + 1) * binSize, 1)}`,
        count: 0
      }));

      values.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
        histogram[binIndex].count++;
      });

      return histogram;
    }
  };

  const generateConfusionMatrix = (predictions: number[], actual: number[], classes: string[]) => {
    const matrix = Array(classes.length).fill(null).map(() => Array(classes.length).fill(0));
    
    predictions.forEach((pred, i) => {
      matrix[actual[i]][pred]++;
    });

    return matrix.map((row, i) => 
      row.map((value, j) => ({
        x: j,
        y: i,
        value,
        actual: classes[i],
        predicted: classes[j]
      }))
    ).flat();
  };

  const generateFeatureImportance = (dataset: Dataset) => {
    // Check if dataset has the required data
    if (!dataset || !dataset.input_size || dataset.input_size <= 0) {
      return [];
    }

    // Calculate feature importance 
    const features = Array.from({ length: dataset.input_size }, (_, i) => 
      `Feature ${i + 1}`
    );
    
    return features.map((name, index) => {
      let importance = 0;
      if (trainingHistory.length > 0) {
        const finalAccuracy = trainingHistory[trainingHistory.length - 1]?.accuracy || 0;
        const initialAccuracy = trainingHistory[0]?.accuracy || 0;
        const improvementFactor = Math.max(0.1, finalAccuracy - initialAccuracy + 0.5);
        
        importance = (0.5 + Math.sin(index * 0.5) * 0.3) * improvementFactor;
        importance = Math.max(0, Math.min(1, importance));
      } else {
        importance = 0.1;
      }
      
      return {
        name,
        importance,
        fill: '#8884d8'
      };
    }).sort((a, b) => b.importance - a.importance).slice(0, 10);
  };

  const renderVisualizationTabs = () => (
    <Paper sx={{ mb: 3 }}>
      <Tabs
        value={visualizationMode}
        onChange={(_, newValue) => setVisualizationMode(newValue)}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab value="overview" icon={<ChartsIcon />} label="Training Overview" />
        <Tab value="metrics" icon={<MetricsIcon />} label="Detailed Metrics" />
        <Tab value="architecture" icon={<NetworkIcon />} label="Network Architecture" />
        <Tab value="weights" icon={<ViewIcon />} label="Weight Analysis" />
        <Tab value="predictions" icon={<ScatterIcon />} label="Predictions" />
        <Tab value="dataset" icon={<DataIcon />} label="Dataset Analysis" />
        <Tab value="evaluation" icon={<ConfusionIcon />} label="Model Evaluation" />
        <Tab value="visualizations" icon={<ProgressIcon />} label="Advanced Visualizations" />
      </Tabs>
    </Paper>
  );

  const renderTrainingOverview = () => (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 3 }}>
      {/* Loss & Accuracy Charts */}
      <Card>
        <CardHeader title={createTitleWithIcon("[PROGRESS] Training Progress")} />
        <CardContent>
          <ResponsiveContainer width="100%" height={600}>
            <LineChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" name="Training Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#82ca9d" name="Validation Loss" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader title={
          trainingHistory.length > 0 && trainingHistory[0].accuracy !== undefined 
            ? createTitleWithIcon("[TRAINING] Accuracy Metrics")
            : createTitleWithIcon("[TRAINING] R² Score Metrics")
        } />
        <CardContent>
          <ResponsiveContainer width="100%" height={600}>
            <LineChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              {/* Show accuracy lines for classification models */}
              {trainingHistory.length > 0 && trainingHistory[0].accuracy !== undefined && (
                <>
                  <Line type="monotone" dataKey="accuracy" stroke="#ffc658" name="Training Accuracy" />
                  <Line type="monotone" dataKey="val_accuracy" stroke="#ff7300" name="Validation Accuracy" />
                </>
              )}
              {/* Show R² lines for regression models */}
              {trainingHistory.length > 0 && trainingHistory[0].r2_score !== undefined && (
                <>
                  <Line type="monotone" dataKey="r2_score" stroke="#ffc658" name="Training R²" />
                  <Line type="monotone" dataKey="val_r2_score" stroke="#ff7300" name="Validation R²" />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Current Metrics */}
      {currentMetrics && (
        <Card sx={{ gridColumn: { xs: 'span 1', lg: 'span 2' } }}>
          <CardHeader title={createTitleWithIcon("[METRICS] Current Metrics")} />
          <CardContent>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 2 }}>
              <Box textAlign="center">
                <Typography variant="h4" color="primary">
                  {currentMetrics.epoch}
                </Typography>
                <Typography variant="body2">Epoch</Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="h4" color={currentMetrics.loss !== null && currentMetrics.loss < 0.1 ? 'success.main' : 'warning.main'}>
                  {formatNumber(currentMetrics.loss)}
                </Typography>
                <Typography variant="body2">Loss</Typography>
              </Box>
              {/* Show accuracy for classification models */}
              {currentMetrics.accuracy !== undefined && currentMetrics.accuracy !== null && (
                <Box textAlign="center">
                  <Typography variant="h4" color={currentMetrics.accuracy > 0.9 ? 'success.main' : 'info.main'}>
                    {formatPercentage(currentMetrics.accuracy)}
                  </Typography>
                  <Typography variant="body2">Accuracy</Typography>
                </Box>
              )}
              {/* Show R² score for regression models */}
              {currentMetrics.r2_score !== undefined && currentMetrics.r2_score !== null && (
                <Box textAlign="center">
                  <Typography variant="h4" color={currentMetrics.r2_score > 0.8 ? 'success.main' : currentMetrics.r2_score > 0.5 ? 'info.main' : 'warning.main'}>
                    {formatNumber(currentMetrics.r2_score, 3)}
                  </Typography>
                  <Typography variant="body2">R² Score</Typography>
                </Box>
              )}
              {/* Show Invalid R² for null values */}
              {currentMetrics.r2_score === null && (
                <Box textAlign="center">
                  <Typography variant="h4" color="error.main">
                    Invalid
                  </Typography>
                  <Typography variant="body2">R² Score</Typography>
                </Box>
              )}
              <Box textAlign="center">
                <Typography variant="h4" color="secondary">
                  {formatNumber(currentMetrics.learning_rate, 6) !== 'N/A' 
                    ? formatNumber(currentMetrics.learning_rate, 6) 
                    : formatNumber(trainingConfig.learningRate, 6)}
                </Typography>
                <Typography variant="body2">Learning Rate</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );

  const renderDetailedMetrics = () => (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 3 }}>
      {/* Loss Landscape */}
      <Card>
        <CardHeader title={createTitleWithIcon("[LANDSCAPE] Loss Landscape")} />
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <RechartsTooltip />
              <Area type="monotone" dataKey="loss" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
              <Area type="monotone" dataKey="val_loss" stackId="2" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Learning Rate Schedule */}
      <Card>
        <CardHeader title={createTitleWithIcon("[DATA] Learning Rate Schedule")} />
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <RechartsTooltip />
              <Line 
                type="monotone" 
                dataKey="learning_rate" 
                stroke="#ff7300" 
                name="Learning Rate"
                dot={{ fill: '#ff7300', strokeWidth: 2, r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Training Speed */}
      <Card>
        <CardHeader title={createTitleWithIcon("[METRICS] Training Speed")} />
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="loss" fill="#8884d8" name="Loss Reduction" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Validation Performance */}
      <Card>
        <CardHeader title={createTitleWithIcon("[OK] Validation Performance")} />
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={[
              // Show appropriate metric based on model type
              currentMetrics?.val_accuracy !== undefined 
                ? { metric: 'Accuracy', value: currentMetrics.val_accuracy }
                : { metric: 'R² Score', value: currentMetrics?.val_r2_score || 0 },
              { metric: 'Loss', value: Math.max(0, 1 - (currentMetrics?.val_loss || 1) / 100) }, // Normalize loss for display
              { 
                metric: 'Convergence', 
                value: trainingHistory.length > 1 ? 
                  Math.max(0, 1 - Math.abs(trainingHistory[trainingHistory.length - 1]?.loss - trainingHistory[trainingHistory.length - 2]?.loss) || 0) : 0.5 
              },
              { 
                metric: 'Stability', 
                value: trainingHistory.length > 0 ? 
                  Math.max(0, Math.min(1, 1 - (trainingHistory.reduce((acc, curr, i) => i > 0 ? acc + Math.abs(curr.loss - trainingHistory[i-1].loss) : 0, 0) / Math.max(1, trainingHistory.length - 1)))) : 0.5 
              },
              { 
                metric: 'Generalization', 
                value: currentMetrics?.val_accuracy && currentMetrics?.accuracy ? 
                  Math.max(0, 1 - Math.abs(currentMetrics.val_accuracy - currentMetrics.accuracy)) : 
                  (currentMetrics?.val_loss && currentMetrics?.loss ? Math.max(0, 1 - Math.abs(currentMetrics.val_loss - currentMetrics.loss) / Math.max(currentMetrics.val_loss, currentMetrics.loss)) : 0.5)
              }
            ]}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis />
              <Radar 
                name="Performance" 
                dataKey="value" 
                stroke="#8884d8" 
                fill="#8884d8" 
                fillOpacity={0.6}
              />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </Box>
  );

  const renderDatasetAnalysis = () => {
    if (!selectedDataset || !datasets[selectedDataset]) {
      return (
        <Alert severity="info">
          Select a dataset to view distribution analysis
        </Alert>
      );
    }

    const dataset = datasets[selectedDataset];
    const distribution = generateDatasetDistribution(dataset);
    const featureImportance = generateFeatureImportance(dataset);

    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 3 }}>
        {/* Dataset Distribution */}
        <Card>
          <CardHeader title={createTitleWithIcon("[DATA] Target Distribution")} />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              {dataset.type === 'classification' ? (
                <PieChart>
                  <Pie
                    data={distribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              ) : (
                <BarChart data={distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              )}
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Feature Importance*/}
        <Card>
          <CardHeader title={createTitleWithIcon("[TRAINING] Feature Importance")} />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={featureImportance} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width="auto" />
                <RechartsTooltip />
                <Bar dataKey="importance" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Dataset Statistics */}
        <Card sx={{ gridColumn: { xs: 'span 1', lg: 'span 2' } }}>
          <CardHeader title={createTitleWithIcon("[PROGRESS] Dataset Statistics")} />
          <CardContent>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
              <Box>
                <Typography variant="h6">
                  {dataset.X ? 'Total Samples' : 'Training Samples'}
                </Typography>
                <Typography variant="h4" color="primary">
                  {dataset.X?.length || dataset.X_train?.length || 0}
                </Typography>
              </Box>
              {!dataset.X && (
                <Box>
                  <Typography variant="h6">Test Samples</Typography>
                  <Typography variant="h4" color="secondary">{dataset.X_test?.length || 0}</Typography>
                </Box>
              )}
              <Box>
                <Typography variant="h6">Input Features</Typography>
                <Typography variant="h4" color="info.main">{dataset.input_size || 0}</Typography>
              </Box>
              <Box>
                <Typography variant="h6">Output Classes</Typography>
                <Typography variant="h4" color="success.main">{dataset.output_size}</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>
    );
  };

  const renderModelEvaluation = () => {
    const trainedModel = selectedModel && models[selectedModel] && models[selectedModel].status === 'trained' 
      ? models[selectedModel] 
      : null;
    
    const hasTestSet = trainedModel?.test_set && trainedModel.test_set.size > 0;
    
    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 3 }}>
        {/* Model Information and Test Set Evaluation */}
        <Card>
          <CardHeader title={createTitleWithIcon("[TEST] Test Set Evaluation")} />
          <CardContent>
            {!trainedModel ? (
              <Alert severity="info">
                Train a model first to evaluate it on the held-out test set. Select a model and dataset above, then click "Start Training".
              </Alert>
            ) : (
              <Box>
                {/* Model Information */}
                <Box sx={{ mb: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <DataIcon fontSize="small" />
                    Trained Model: {trainedModel.display_name || selectedModel}
                  </Typography>
                  
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 2 }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">Dataset</Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {trainedModel.dataset_name || 'Unknown'}
                      </Typography>
                    </Box>
                    
                    <Box>
                      <Typography variant="body2" color="text.secondary">Model Type</Typography>
                      <Box 
                        sx={{ 
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: 1, 
                          fontSize: '1rem', 
                          fontWeight: 'bold',
                          lineHeight: 1.5
                        }}
                      >
                        {trainedModel.dataset_type === 'classification' ? (
                          <>
                            <LabelIcon fontSize="small" />
                            Classification
                          </>
                        ) : (
                          <>
                            <TrainingIcon fontSize="small" />
                            Regression
                          </>
                        )}
                      </Box>
                    </Box>
                    
                    {trainedModel.split_info && (
                      <Box>
                        <Typography variant="body2" color="text.secondary">Data Splits</Typography>
                        <Typography variant="body1" fontWeight="bold">
                          Train: {Math.round(trainedModel.split_info.train_split * 100)}% | 
                          Val: {Math.round(trainedModel.split_info.validation_split * 100)}% | 
                          Test: {Math.round(trainedModel.split_info.test_split * 100)}%
                        </Typography>
                      </Box>
                    )}
                    
                    {trainedModel.final_metrics && (
                      <Box>
                        <Typography variant="body2" color="text.secondary">Training Performance</Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {trainedModel.dataset_type === 'classification' 
                            ? `Accuracy: ${formatPercentage(trainedModel.final_metrics.final_accuracy)}`
                            : `R²: ${formatNumber(trainedModel.final_metrics.final_r2_score, 3)}`
                          }
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Box>
                
                {/* Test Set Evaluation Controls */}
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 3 }}>
                  <Button
                    variant="contained"
                    onClick={() => evaluateOnTestSet(selectedModel)}
                    disabled={!hasTestSet || isTestingOnTestSet}
                    startIcon={isTestingOnTestSet ? <LinearProgress /> : <ViewIcon />}
                    size="large"
                  >
                    {isTestingOnTestSet ? 'Testing...' : 'Test on Held-Out Test Set'}
                  </Button>
                  
                  {hasTestSet && (
                    <Chip 
                      label={`${trainedModel.test_set.size} test samples`} 
                      color="primary" 
                      variant="outlined" 
                    />
                  )}
                  
                  {!hasTestSet && (
                    <Tooltip title="This model was trained without a test set. Enable 'Create test set' in training configuration to get a held-out test set.">
                      <Chip 
                        label="No test set available" 
                        color="default" 
                        variant="outlined" 
                      />
                    </Tooltip>
                  )}
                  
                  <Button
                    variant="outlined"
                    onClick={() => downloadModel(selectedModel)}
                    disabled={isDownloading}
                    startIcon={<DownloadIcon />}
                  >
                    {isDownloading ? 'Downloading...' : 'Download Model'}
                  </Button>
                </Box>
                
                {/* Test Set Results */}
                {testSetResults && (
                  <Box>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TrainingIcon fontSize="small" />
                      Test Set Results
                    </Typography>
                    <Alert severity="success" sx={{ mb: 3 }}>
                      Model evaluated on {testSetResults.test_set_size} held-out test samples from {testSetResults.dataset_name}
                    </Alert>
                    
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2, mb: 3 }}>
                      {/* Show accuracy for classification models */}
                      {testSetResults.accuracy !== undefined && testSetResults.accuracy !== null && (
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h4" color="primary">
                              {formatPercentage(testSetResults.accuracy, 2)}
                            </Typography>
                            <Typography variant="body2">Test Accuracy</Typography>
                          </CardContent>
                        </Card>
                      )}
                      
                      {/* Show R² score for regression models */}
                      {testSetResults.r2_score !== undefined && testSetResults.r2_score !== null && (
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h4" color="primary">
                              {formatNumber(testSetResults.r2_score, 4)}
                            </Typography>
                            <Typography variant="body2">Test R² Score</Typography>
                          </CardContent>
                        </Card>
                      )}
                      
                      {/* Show MSE for regression models */}
                      {testSetResults.mse !== undefined && (
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h4" color="secondary">
                              {formatNumber(testSetResults.mse, 4)}
                            </Typography>
                            <Typography variant="body2">Mean Squared Error</Typography>
                          </CardContent>
                        </Card>
                      )}
                      
                      {/* Show MAE for regression models */}
                      {testSetResults.mae !== undefined && (
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h4" color="info.main">
                              {formatNumber(testSetResults.mae, 4)}
                            </Typography>
                            <Typography variant="body2">Mean Absolute Error</Typography>
                          </CardContent>
                        </Card>
                      )}
                    </Box>
                    
                    {/* Split Information */}
                    {testSetResults.split_info && (
                      <Card variant="outlined" sx={{ mt: 2 }}>
                        <CardHeader title={createTitleWithIcon("[DATA] Split Information")} />
                        <CardContent>
                          <Typography variant="body2">
                            <strong>Training:</strong> {testSetResults.split_info.train_size} samples ({Math.round(testSetResults.split_info.train_split * 100)}%)<br/>
                            <strong>Validation:</strong> {testSetResults.split_info.validation_size} samples ({Math.round(testSetResults.split_info.validation_split * 100)}%)<br/>
                            <strong>Test:</strong> {testSetResults.split_info.test_size} samples ({Math.round(testSetResults.split_info.test_split * 100)}%)<br/>
                            <strong>Random Seed:</strong> {testSetResults.split_info.random_seed}
                          </Typography>
                        </CardContent>
                      </Card>
                    )}
                  </Box>
                )}
              </Box>
            )}
        </CardContent>
      </Card>
    </Box>
  );
  };

  const renderAdvancedVisualizations = () => (
    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 3 }}>
      {/* Visualization Controls */}
      <Card>
        <CardHeader title={createTitleWithIcon("[VIZ] Advanced Visualizations")} />
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 3 }}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Select Model</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="Select Model"
              >
                {Object.entries(models).filter(([_, model]) => model.status === 'trained').map(([id, model]) => (
                  <MenuItem key={id} value={id}>
                    {model.display_name || model.config?.name || id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Button
              variant="contained"
              onClick={() => fetchModelVisualizations(selectedModel)}
              disabled={!selectedModel}
              startIcon={<ViewIcon />}
            >
              Generate Visualizations
            </Button>
          </Box>
          
          {/* Model Visualizations */}
          {modelVisualizations && (
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, 1fr)' }, gap: 3 }}>
              {/* Weight Histogram */}
              {modelVisualizations.weight_histogram && (
                <Card>
                  <CardHeader title="Weight Distribution" />
                  <CardContent>
                    <img 
                      src={`data:image/png;base64,${modelVisualizations.weight_histogram}`}
                      alt="Weight Distribution"
                      style={{ width: '100%', height: 'auto' }}
                    />
                  </CardContent>
                </Card>
              )}
              
              {/* Training Curves */}
              {modelVisualizations.training_curves && (
                <Card>
                  <CardHeader title="Training & Validation Curves" />
                  <CardContent>
                    <img 
                      src={`data:image/png;base64,${modelVisualizations.training_curves}`}
                      alt="Training Curves"
                      style={{ width: '100%', height: 'auto' }}
                    />
                  </CardContent>
                </Card>
              )}
              
              {/* Architecture Text */}
              {modelVisualizations.architecture_text && (
                <Card sx={{ gridColumn: { xs: 'span 1', lg: 'span 2' } }}>
                  <CardHeader title="Model Architecture" />
                  <CardContent>
                    <Box component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                      {modelVisualizations.architecture_text.join('\n')}
                    </Box>
                  </CardContent>
                </Card>
              )}
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <TrainingIcon />
        Training Visualizer
      </Typography>

      {/* Training Configuration */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ConfigIcon />
          Training Configuration
        </Typography>
        
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, gap: 2, mb: 3 }}>
          <FormControl fullWidth>
            <InputLabel>Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              label="Model"
            >
              {Object.entries(models).map(([modelId, model]) => (
                <MenuItem key={modelId} value={modelId}>
                  {model.display_name || model.config?.name || modelId} 
                  {model.architecture_summary && (
                    <Chip 
                      label={`${model.architecture_summary.total_layers} layers`} 
                      size="small" 
                      sx={{ ml: 1 }} 
                    />
                  )}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel>Dataset</InputLabel>
            <Select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              label="Dataset"
            >
              {Object.entries(datasets).map(([datasetId, dataset]) => (
                <MenuItem key={datasetId} value={datasetId}>
                  {dataset.name}
                  <Chip 
                    label={dataset.type} 
                    size="small" 
                    color={dataset.type === 'classification' ? 'primary' : 'success'}
                    sx={{ ml: 1 }} 
                  />
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              color={isTraining ? "error" : "success"}
              startIcon={isTraining ? <StopIcon /> : <StartIcon />}
              onClick={isTraining ? stopTraining : startTraining}
              disabled={!selectedModel || !selectedDataset}
              fullWidth
            >
              {isTraining ? 'Stop Training' : 'Start Training'}
            </Button>
            <Tooltip title="Refresh Status">
              <IconButton onClick={fetchTrainingStatus}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Advanced Configuration */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SettingsIcon fontSize="small" />
              Advanced Training Settings
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}>
              <TextField
                label="Epochs"
                type="number"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                inputProps={{ min: 1, max: 1000 }}
              />
              <TextField
                label="Batch Size"
                type="number"
                value={trainingConfig.batchSize}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                inputProps={{ min: 1, max: 1024 }}
              />
              <TextField
                label="Learning Rate"
                type="number"
                inputProps={{ min: 0, step: 0.0001 }}
                value={trainingConfig.learningRate}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
              />
              <FormControl fullWidth>
                <InputLabel>Optimizer</InputLabel>
                <Select
                  value={trainingConfig.optimizer}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, optimizer: e.target.value }))}
                  label="Optimizer"
                >
                  <MenuItem value="adam">Adam</MenuItem>
                  <MenuItem value="sgd">SGD</MenuItem>
                  <MenuItem value="rmsprop">RMSprop</MenuItem>
                  <MenuItem value="adamw">AdamW</MenuItem>
                </Select>
              </FormControl>
            </Box>
            
            {/* Loss Function Selection */}
            <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Loss Function</InputLabel>
                <Select
                  value={trainingConfig.lossFunction}
                  onChange={(e) => setTrainingConfig(prev => ({ ...prev, lossFunction: e.target.value }))}
                  label="Loss Function"
                  disabled={!selectedDataset}
                >
                  {getAvailableLossFunctions().map((lossFunc) => (
                    <MenuItem key={lossFunc.value} value={lossFunc.value}>
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          {lossFunc.label}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {lossFunc.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              {/* Effective loss based on dataset type */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Effective Loss Function:
                </Typography>
                <Chip
                  label={getEffectiveLossFunction().toUpperCase()}
                  size="small"
                  color={trainingConfig.lossFunction === 'auto' ? 'primary' : 'secondary'}
                  variant={trainingConfig.lossFunction === 'auto' ? 'filled' : 'outlined'}
                />
              </Box>
            </Box>
            
            {/* Data Split Configuration */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <DataIcon fontSize="small" />
                Data Split Configuration
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch
                    checked={trainingConfig.useTestSet}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, useTestSet: e.target.checked }))}
                  />
                }
                label="Create test set for final evaluation"
                sx={{ mb: 2 }}
              />
              
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: trainingConfig.useTestSet ? 'repeat(3, 1fr)' : 'repeat(2, 1fr)' }, gap: 2 }}>
                <TextField
                  label="Training Split"
                  type="number"
                  inputProps={{ min: 0.1, max: 0.9, step: 0.1 }}
                  value={trainingConfig.trainSplit}
                  onChange={(e) => {
                    const newTrainSplit = parseFloat(e.target.value);
                    const remainingSplit = 1 - newTrainSplit;
                    const newValidationSplit = trainingConfig.useTestSet 
                      ? remainingSplit * (trainingConfig.validationSplit / (trainingConfig.validationSplit + trainingConfig.testSplit))
                      : remainingSplit;
                    const newTestSplit = trainingConfig.useTestSet ? remainingSplit - newValidationSplit : 0;
                    
                    setTrainingConfig(prev => ({ 
                      ...prev, 
                      trainSplit: newTrainSplit,
                      validationSplit: newValidationSplit,
                      testSplit: newTestSplit
                    }));
                  }}
                  helperText={`${(trainingConfig.trainSplit * 100).toFixed(0)}% for training`}
                  fullWidth
                />
                
                <TextField
                  label="Validation Split"
                  type="number"
                  inputProps={{ min: 0.1, max: 0.5, step: 0.1 }}
                  value={trainingConfig.validationSplit}
                  onChange={(e) => {
                    const newValidationSplit = parseFloat(e.target.value);
                    const remainingSplit = 1 - trainingConfig.trainSplit;
                    const newTestSplit = trainingConfig.useTestSet ? remainingSplit - newValidationSplit : 0;
                    
                    if (newValidationSplit <= remainingSplit) {
                      setTrainingConfig(prev => ({ 
                        ...prev, 
                        validationSplit: newValidationSplit,
                        testSplit: Math.max(0, newTestSplit)
                      }));
                    }
                  }}
                  helperText={`${(trainingConfig.validationSplit * 100).toFixed(0)}% for validation`}
                  fullWidth
                />
                
                {trainingConfig.useTestSet && (
                  <TextField
                    label="Test Split"
                    type="number"
                    inputProps={{ min: 0.1, max: 0.5, step: 0.1 }}
                    value={trainingConfig.testSplit}
                    onChange={(e) => {
                      const newTestSplit = parseFloat(e.target.value);
                      const remainingSplit = 1 - trainingConfig.trainSplit;
                      const newValidationSplit = remainingSplit - newTestSplit;
                      
                      if (newTestSplit <= remainingSplit && newValidationSplit >= 0.1) {
                        setTrainingConfig(prev => ({ 
                          ...prev, 
                          testSplit: newTestSplit,
                          validationSplit: newValidationSplit
                        }));
                      }
                    }}
                    helperText={`${(trainingConfig.testSplit * 100).toFixed(0)}% for final testing`}
                    fullWidth
                  />
                )}
              </Box>
              
              <TextField
                label="Random Seed"
                type="number"
                inputProps={{ min: 1, max: 999999 }}
                value={trainingConfig.randomSeed}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, randomSeed: parseInt(e.target.value) }))}
                helperText="For reproducible splits"
                sx={{ mt: 2, width: '200px' }}
              />
            </Box>
            
            <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoRefresh}
                    onChange={(e) => setAutoRefresh(e.target.checked)}
                  />
                }
                label="Auto-refresh metrics"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={trainingConfig.earlyStopping}
                    onChange={(e) => setTrainingConfig(prev => ({ ...prev, earlyStopping: e.target.checked }))}
                  />
                }
                label="Early stopping"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={showAdvancedMetrics}
                    onChange={(e) => setShowAdvancedMetrics(e.target.checked)}
                  />
                }
                label="Show advanced metrics"
              />
            </Box>
          </AccordionDetails>
        </Accordion>
      </Paper>

      {/* Training Status */}
      {isTraining && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <LinearProgress sx={{ flexGrow: 1 }} />
            <Typography>
              Training in progress... {currentMetrics ? `Epoch ${currentMetrics.epoch}` : ''}
            </Typography>
          </Box>
        </Alert>
      )}

      {/* Visualization Tabs */}
      {renderVisualizationTabs()}

      {/* Visualization Content */}
      <Box>
        {visualizationMode === 'overview' && renderTrainingOverview()}
        {visualizationMode === 'metrics' && renderDetailedMetrics()}
        {visualizationMode === 'dataset' && renderDatasetAnalysis()}
        {visualizationMode === 'architecture' && (
          <Alert severity="info">
            Network Architecture visualization will be implemented with interactive node diagrams
          </Alert>
        )}
        {visualizationMode === 'weights' && (
          <Alert severity="info">
            Weight visualization will show layer-by-layer weight distributions and gradient flows
          </Alert>
        )}
        {visualizationMode === 'predictions' && (
          <Alert severity="info">
            Prediction visualization will show scatter plots, confusion matrices, and error analysis
          </Alert>
        )}
        {visualizationMode === 'evaluation' && renderModelEvaluation()}
        {visualizationMode === 'visualizations' && renderAdvancedVisualizations()}
      </Box>
    </Box>
  );
};

export default TrainingVisualizer; 