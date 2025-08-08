import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Psychology as ModelIcon,
  Timeline as MetricsIcon,
  Architecture as ArchitectureIcon,
  PlayArrow as TestIcon
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface Model {
  id: string;
  config: any;
  created_at: string;
  status: 'created' | 'training' | 'trained' | 'failed';
  training_history: TrainingMetric[];
  display_name?: string;
}

interface TrainingMetric {
  epoch: number;
  loss: number;
  accuracy?: number;        // For classification models
  val_accuracy?: number;    // For classification models
  r2_score?: number;        // For regression models
  val_r2_score?: number;    // For regression models
  timestamp: string;
}

// Safe number formatting function to handle null/undefined/NaN values
const formatNumber = (num: number | null | undefined, decimals: number = 8) => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  return num.toFixed(decimals);
};

const formatPercentage = (num: number | null | undefined, decimals: number = 8) => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  return (num * 100).toFixed(decimals) + '%';
};

// Helper function to determine if model is regression (has r2_score) or classification (has accuracy)
const isRegressionModel = (trainingHistory: TrainingMetric[]): boolean => {
  if (trainingHistory.length === 0) return false;
  const firstMetric = trainingHistory[0];
  return firstMetric.r2_score !== undefined;
};

// Helper function to get the appropriate metric value (accuracy for classification, r2_score for regression)
const getModelMetricValue = (metric: TrainingMetric, isRegression: boolean): number | undefined => {
  return isRegression ? metric.r2_score : metric.accuracy;
};

// Helper function to get the metric name for display
const getMetricName = (isRegression: boolean): string => {
  return isRegression ? 'R² Score' : 'Accuracy';
};

const ModelManager: React.FC = () => {
  const [models, setModels] = useState<Record<string, Model>>({});
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);
  const [viewType, setViewType] = useState<'overview' | 'metrics' | 'architecture'>('overview');

  useEffect(() => {
    fetchModels();
    // Set up polling for model updates
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/models');
      const data = await response.json();
      setModels(data);
    } catch (error) {

    } finally {
      setLoading(false);
    }
  };

  const viewModelDetails = (model: Model) => {
    setSelectedModel(model);
    setViewType('overview');
    setDetailsDialogOpen(true);
  };

  const deleteModel = async (modelId: string) => {
    if (window.confirm('Are you sure you want to delete this model?')) {
      // In a real implementation, this would call the backend to delete the model

      // For now, just remove from local state
      setModels(prev => {
        const newModels = { ...prev };
        delete newModels[modelId];
        return newModels;
      });
    }
  };

  const downloadModel = (modelId: string) => {
    // In a real implementation, this would download the model weights/configuration

    alert('Model download functionality would be implemented here');
  };

  const testModel = (modelId: string) => {
    // In a real implementation, this would run inference on test data

    alert('Model testing functionality would be implemented here');
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'created': return 'default';
      case 'training': return 'warning';
      case 'trained': return 'success';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const getLayerSummary = (config: any) => {
    if (!config.layers) return 'No layers configured';
    return `${config.layers.length} layers`;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Model Manager
        </Typography>
        <Button variant="outlined" onClick={fetchModels}>
          Refresh Models
        </Button>
      </Box>

      {Object.keys(models).length === 0 ? (
        <Alert severity="info">
          No models found. Create a model in the Network Builder tab to get started.
        </Alert>
      ) : (
                 <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 3 }}>
           {Object.entries(models).map(([modelId, model]) => (
             <Box key={modelId}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <ModelIcon sx={{ mr: 1 }} />
                    <Typography variant="h6" component="div">
                      {model.display_name || model.config?.name || modelId}
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                    <Chip 
                      label={model.status} 
                      color={getStatusColor(model.status) as any}
                      size="small"
                    />
                    <Chip 
                      label={getLayerSummary(model.config)} 
                      variant="outlined" 
                      size="small"
                    />
                    {model.training_history.length > 0 && (
                      <Chip 
                        label={`${model.training_history.length} epochs`} 
                        variant="outlined" 
                        size="small"
                      />
                    )}
                  </Box>

                  <Typography variant="body2" color="text.secondary">
                    Created: {new Date(model.created_at).toLocaleDateString()}
                  </Typography>

                  {model.status === 'training' && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" gutterBottom>
                        Training in progress...
                      </Typography>
                      <LinearProgress />
                    </Box>
                  )}

                  {model.status === 'trained' && model.training_history.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Final Loss: {formatNumber(model.training_history[model.training_history.length - 1].loss)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Final {getMetricName(isRegressionModel(model.training_history))}: {isRegressionModel(model.training_history) 
                          ? formatNumber(getModelMetricValue(model.training_history[model.training_history.length - 1], true))
                          : formatPercentage(getModelMetricValue(model.training_history[model.training_history.length - 1], false))
                        }
                      </Typography>
                    </Box>
                  )}
                </CardContent>
                
                <CardActions>
                  <Button 
                    size="small" 
                    startIcon={<ViewIcon />}
                    onClick={() => viewModelDetails(model)}
                  >
                    Details
                  </Button>
                  
                  {model.status === 'trained' && (
                    <>
                      <Tooltip title="Test Model">
                        <IconButton 
                          size="small" 
                          onClick={() => testModel(modelId)}
                          color="primary"
                        >
                          <TestIcon />
                        </IconButton>
                      </Tooltip>
                      
                      <Tooltip title="Download Model">
                        <IconButton 
                          size="small" 
                          onClick={() => downloadModel(modelId)}
                          color="primary"
                        >
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                    </>
                  )}
                  
                  <Tooltip title="Delete Model">
                    <IconButton 
                      size="small" 
                      onClick={() => deleteModel(modelId)}
                      color="error"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </CardActions>
              </Card>
            </Box>
          ))}
        </Box>
      )}

      {/* Model Details Dialog */}
      <Dialog 
        open={detailsDialogOpen} 
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <ModelIcon sx={{ mr: 1 }} />
              Model Details: {selectedModel && Object.keys(models).find(key => models[key] === selectedModel)}
            </Box>
            <Box>
              <Button
                variant={viewType === 'overview' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewType('overview')}
                sx={{ mr: 1 }}
              >
                Overview
              </Button>
              <Button
                variant={viewType === 'metrics' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewType('metrics')}
                sx={{ mr: 1 }}
                disabled={!selectedModel?.training_history.length}
              >
                Metrics
              </Button>
              <Button
                variant={viewType === 'architecture' ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setViewType('architecture')}
              >
                Architecture
              </Button>
            </Box>
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {selectedModel && (
            <Box>
              {viewType === 'overview' && (
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                  <Box>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        General Information
                      </Typography>
                      <Table size="small">
                        <TableBody>
                          <TableRow>
                            <TableCell>Status</TableCell>
                            <TableCell>
                              <Chip 
                                label={selectedModel.status} 
                                color={getStatusColor(selectedModel.status) as any}
                                size="small"
                              />
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Created</TableCell>
                            <TableCell>{new Date(selectedModel.created_at).toLocaleString()}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Layers</TableCell>
                            <TableCell>{getLayerSummary(selectedModel.config)}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Training Epochs</TableCell>
                            <TableCell>{selectedModel.training_history.length}</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </Paper>
                  </Box>
                  
                  {selectedModel.status === 'trained' && selectedModel.training_history.length > 0 && (
                    <Box>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                          Training Summary
                        </Typography>
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell>Final Loss</TableCell>
                              <TableCell>
                                {formatNumber(selectedModel.training_history[selectedModel.training_history.length - 1].loss)}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Final {getMetricName(isRegressionModel(selectedModel.training_history))}</TableCell>
                              <TableCell>
                                {isRegressionModel(selectedModel.training_history) 
                                  ? formatNumber(getModelMetricValue(selectedModel.training_history[selectedModel.training_history.length - 1], true))
                                  : formatPercentage(getModelMetricValue(selectedModel.training_history[selectedModel.training_history.length - 1], false))
                                }
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Best Loss</TableCell>
                              <TableCell>
                                {formatNumber(Math.min(...selectedModel.training_history.map(h => h.loss)))}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>Best {getMetricName(isRegressionModel(selectedModel.training_history))}</TableCell>
                              <TableCell>
                                {isRegressionModel(selectedModel.training_history) 
                                  ? formatNumber(Math.max(...selectedModel.training_history.map(h => getModelMetricValue(h, true) || -Infinity).filter(v => v !== -Infinity)))
                                  : formatPercentage(Math.max(...selectedModel.training_history.map(h => getModelMetricValue(h, false) || -Infinity).filter(v => v !== -Infinity)))
                                }
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </Paper>
                    </Box>
                  )}
                </Box>
              )}

              {viewType === 'metrics' && selectedModel.training_history.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Training Metrics
                  </Typography>
                  <Box sx={{ height: 400, mb: 3 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={selectedModel.training_history}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="epoch" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <RechartsTooltip />
                        <Legend />
                        <Line 
                          yAxisId="left"
                          type="monotone" 
                          dataKey="loss" 
                          stroke="#f44336" 
                          strokeWidth={2}
                          name="Loss"
                        />
                        <Line 
                          yAxisId="right"
                          type="monotone" 
                          dataKey="accuracy" 
                          stroke="#4caf50" 
                          strokeWidth={2}
                          name="Accuracy"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Box>
              )}

              {viewType === 'architecture' && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Network Architecture
                  </Typography>
                  {selectedModel.config.layers ? (
                    <TableContainer component={Paper} variant="outlined">
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Layer #</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Parameters</TableCell>
                            <TableCell>Activation</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {selectedModel.config.layers.map((layer: any, index: number) => (
                            <TableRow key={index}>
                              <TableCell>{index + 1}</TableCell>
                              <TableCell>
                                <Chip 
                                  label={layer.layer_type} 
                                  size="small"
                                  variant="outlined"
                                />
                              </TableCell>
                              <TableCell>
                                {Object.entries(layer.params).map(([key, value]) => (
                                  <Chip 
                                    key={key}
                                    label={`${key}: ${value}`} 
                                    size="small"
                                    variant="outlined"
                                    sx={{ mr: 0.5, mb: 0.5 }}
                                  />
                                ))}
                              </TableCell>
                              <TableCell>
                                {layer.activation && (
                                  <Chip 
                                    label={layer.activation} 
                                    size="small"
                                    color="primary"
                                  />
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  ) : (
                    <Alert severity="info">
                      No layer configuration available for this model.
                    </Alert>
                  )}
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
          {selectedModel?.status === 'trained' && (
            <>
              <Button variant="outlined" onClick={() => testModel('current')}>
                Test Model
              </Button>
              <Button variant="contained" onClick={() => downloadModel('current')}>
                Download
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ModelManager; 