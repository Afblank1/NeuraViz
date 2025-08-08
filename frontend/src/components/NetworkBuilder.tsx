import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Card,
  CardContent,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  Alert,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Settings as SettingsIcon,
  PlayArrow as PlayIcon,
  Save as SaveIcon,
  ExpandMore as ExpandMoreIcon,
  AddCircle as AddBeforeIcon,
  AddCircleOutline as AddAfterIcon,
  Warning as WarningIcon,
  ViewModule as TemplatesIcon,
  Architecture as NetworkIcon,
  ControlCamera as ControlsIcon,
  BarChart as ChartIcon,
  BugReport as IssuesIcon,
  CheckCircle as ValidIcon,
  Tune as ParamsIcon,
  Functions as ActivationIcon,
  Engineering as AdvancedIcon,
  Loop as LoopIcon,
  Visibility as AttentionIcon
} from '@mui/icons-material';

interface LayerConfig {
  id: string;
  type: string;
  name: string;
  params: Record<string, any>;
  activation?: string;
  dropout_rate?: number;
  batch_norm?: boolean;
  learning_rate?: number;
  weight_init?: string;
  use_bias?: boolean;
  // CNN specific
  kernel_size?: number;
  stride?: number;
  padding?: string | number;
}

const LAYER_TYPES = [
  { type: 'dense', name: 'Dense/Linear Layer', color: '#2196f3', icon: <NetworkIcon fontSize="small" /> },
  { type: 'conv2d', name: 'Convolutional 2D', color: '#4caf50', icon: <ChartIcon fontSize="small" /> },
  { type: 'maxpool2d', name: 'Max Pooling 2D', color: '#ff9800', icon: <ChartIcon fontSize="small" /> },
  { type: 'avgpool2d', name: 'Average Pooling 2D', color: '#ff7043', icon: <ChartIcon fontSize="small" /> },
  { type: 'flatten', name: 'Flatten', color: '#607d8b', icon: <NetworkIcon fontSize="small" /> },
  { type: 'lstm', name: 'LSTM', color: '#795548', icon: <LoopIcon fontSize="small" /> },
  { type: 'attention', name: 'Attention', color: '#e91e63', icon: <AttentionIcon fontSize="small" /> },
];

const ACTIVATION_FUNCTIONS = [
  'none', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'swish', 'gelu', 'softmax', 'softplus'
];

const WEIGHT_INITIALIZATIONS = [
  'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'lecun_uniform', 'lecun_normal', 'random_uniform', 'random_normal', 'zeros', 'ones'
];

const PRESET_TEMPLATES = [
  {
    name: 'Basic MLP',
    description: 'Simple multilayer perceptron for classification',
    layers: [
      { type: 'dense', params: { units: 128 }, activation: 'relu', dropout_rate: 0.3 },
      { type: 'dense', params: { units: 64 }, activation: 'relu' },
      { type: 'dense', params: { units: 10 }, activation: 'softmax' }
    ]
  },
  {
    name: 'CNN for Image Classification',
    description: 'Convolutional network for image recognition',
    layers: [
      { type: 'conv2d', params: { filters: 32, kernel_size: 3 }, activation: 'relu', batch_norm: true },
      { type: 'maxpool2d', params: { pool_size: 2 } },
      { type: 'conv2d', params: { filters: 64, kernel_size: 3 }, activation: 'relu', batch_norm: true },
      { type: 'maxpool2d', params: { pool_size: 2 } },
      { type: 'flatten', params: {} },
      { type: 'dense', params: { units: 128 }, activation: 'relu', dropout_rate: 0.5 },
      { type: 'dense', params: { units: 10 }, activation: 'softmax' }
    ]
  },
  {
    name: 'Deep Regression Network',
    description: 'Deep network for regression tasks',
    layers: [
      { type: 'dense', params: { units: 256 }, activation: 'relu', batch_norm: true, dropout_rate: 0.2 },
      { type: 'dense', params: { units: 128 }, activation: 'relu', batch_norm: true, dropout_rate: 0.2 },
      { type: 'dense', params: { units: 64 }, activation: 'relu' },
      { type: 'dense', params: { units: 1 }, activation: 'none' }
    ]
  }
];

const NetworkBuilder: React.FC = () => {
  const [layers, setLayers] = useState<LayerConfig[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<LayerConfig | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [networkName, setNetworkName] = useState('My Neural Network');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const validateNetwork = useCallback((networkLayers: LayerConfig[]) => {
    const errors: string[] = [];
    
    if (networkLayers.length === 0) {
      errors.push('Network must have at least one layer');
    }
    
    // Check for input layer
    if (networkLayers.length > 0) {
      const firstLayer = networkLayers[0];
      if (firstLayer.type === 'dense' && !firstLayer.params.input_size) {
        errors.push('First dense layer must specify input size');
      } else if (firstLayer.type === 'lstm' && !firstLayer.params.input_size) {
        errors.push('First LSTM layer must specify input size');
      } else if (firstLayer.type === 'conv2d' && (!firstLayer.params.input_height || !firstLayer.params.input_width || !firstLayer.params.input_channels)) {
        errors.push('First conv2d layer must specify input dimensions (height, width, channels)');
      }
    }
    
    // Check CNN sequence
    let expectsImageInput = false;
    networkLayers.forEach((layer, index) => {
      if (layer.type === 'conv2d' || layer.type === 'maxpool2d') {
        expectsImageInput = true;
      } else if (layer.type === 'dense' && expectsImageInput) {
        const prevLayer = networkLayers[index - 1];
        if (prevLayer && prevLayer.type !== 'flatten') {
          errors.push('Dense layer after CNN layers requires Flatten layer');
        }
        expectsImageInput = false;
      }
    });
    
    setValidationErrors(errors);
  }, []);

  const addLayer = useCallback((layerType: string, position?: number) => {
    const newLayer: LayerConfig = {
      id: `layer_${Date.now()}`,
      type: layerType,
      name: LAYER_TYPES.find(lt => lt.type === layerType)?.name || layerType,
      params: getDefaultParams(layerType),
      activation: ['dense', 'conv2d', 'lstm'].includes(layerType) ? 'none' : undefined,
      learning_rate: undefined, // Global by default
      weight_init: ['dense', 'conv2d'].includes(layerType) ? 'xavier_uniform' : undefined,
      use_bias: ['dense', 'conv2d'].includes(layerType) ? true : undefined,
    };
    
    setLayers(prev => {
      const newLayers = [...prev];
      if (position !== undefined) {
        newLayers.splice(position, 0, newLayer);
      } else {
        newLayers.push(newLayer);
      }
      validateNetwork(newLayers);
      return newLayers;
    });
  }, [validateNetwork]);

  const addLayerBefore = useCallback((targetIndex: number, layerType: string) => {
    addLayer(layerType, targetIndex);
  }, [addLayer]);

  const addLayerAfter = useCallback((targetIndex: number, layerType: string) => {
    addLayer(layerType, targetIndex + 1);
  }, [addLayer]);

  const removeLayer = useCallback((layerId: string) => {
    setLayers(prev => {
      const newLayers = prev.filter(layer => layer.id !== layerId);
      validateNetwork(newLayers);
      return newLayers;
    });
  }, [validateNetwork]);

  const loadTemplate = useCallback((template: typeof PRESET_TEMPLATES[0]) => {
    const templateLayers = template.layers.map((layerDef, index) => ({
      id: `template_layer_${index}_${Date.now()}`,
      type: layerDef.type,
      name: LAYER_TYPES.find(lt => lt.type === layerDef.type)?.name || layerDef.type,
      params: layerDef.params,
      ...getDefaultHyperparams(layerDef.type),
      ...layerDef
    }));
    
    setLayers(templateLayers);
    setNetworkName(template.name);
    validateNetwork(templateLayers);
  }, [validateNetwork]);

  const openLayerConfig = useCallback((layer: LayerConfig) => {
    setSelectedLayer(layer);
    setConfigDialogOpen(true);
  }, []);

  const updateLayerConfig = useCallback((updatedLayer: LayerConfig) => {
    setLayers(prev => {
      const newLayers = prev.map(layer => 
        layer.id === updatedLayer.id ? updatedLayer : layer
      );
      validateNetwork(newLayers);
      return newLayers;
    });
    setConfigDialogOpen(false);
    setSelectedLayer(null);
  }, [validateNetwork]);

  const saveNetwork = useCallback(async () => {
    const networkConfig = {
      name: networkName,
      layers: layers.map(layer => ({
        layer_type: layer.type,
        params: layer.params,
        activation: layer.activation,
        dropout_rate: layer.dropout_rate,
        batch_norm: layer.batch_norm,
        learning_rate: layer.learning_rate,
        weight_init: layer.weight_init,
        use_bias: layer.use_bias
      }))
    };
    
    try {
      const response = await fetch('http://localhost:8000/create-network', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(networkConfig)
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Network saved successfully! Model ID: ${result.model_id}`);
      } else {
        alert('Failed to save network');
      }
    } catch (error) {

      alert('Error saving network. Using local storage.');
      localStorage.setItem('neuraviz_network', JSON.stringify(networkConfig));
    }
  }, [layers, networkName]);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Neural Network Builder
      </Typography>
      
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '350px 1fr 300px' }, gap: 3 }}>
        {/* Layer Palette & Templates */}
        <Box>
          {/* Preset Templates */}
          <Paper sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TemplatesIcon fontSize="small" />
              Preset Templates
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {PRESET_TEMPLATES.map((template) => (
                <Button
                  key={template.name}
                  variant="outlined"
                  onClick={() => loadTemplate(template)}
                  sx={{ justifyContent: 'flex-start', textAlign: 'left' }}
                >
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {template.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {template.description}
                    </Typography>
                  </Box>
                </Button>
              ))}
            </Box>
          </Paper>

          {/* Add Layers */}
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              <AddIcon fontSize="small" />
              Available Layers
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {LAYER_TYPES.map((layerType) => (
                <Button
                  key={layerType.type}
                  variant="outlined"
                  startIcon={<span style={{ fontSize: '16px' }}>{layerType.icon}</span>}
                  onClick={() => addLayer(layerType.type)}
                  sx={{ 
                    justifyContent: 'flex-start',
                    borderColor: layerType.color,
                    color: layerType.color,
                    '&:hover': {
                      backgroundColor: `${layerType.color}20`,
                    }
                  }}
                >
                  {layerType.name}
                </Button>
              ))}
            </Box>
          </Paper>
        </Box>

        {/* Network Builder Canvas */}
        <Box>
          <Paper sx={{ p: 2, minHeight: 500 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <NetworkIcon fontSize="small" />
                Network Architecture
              </Typography>
              <TextField
                value={networkName}
                onChange={(e) => setNetworkName(e.target.value)}
                variant="outlined"
                size="small"
                label="Network Name"
              />
            </Box>
            
            {/* Validation Warnings */}
            {validationErrors.length > 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                <Box>
                  <Typography variant="body2" fontWeight="bold" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <WarningIcon fontSize="small" />
                    Network Validation Issues:
                  </Typography>
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    {validationErrors.map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                </Box>
              </Alert>
            )}
            
            {layers.length === 0 ? (
              <Alert severity="info">
                Start building your network by adding layers from the palette on the left, or use a preset template!
              </Alert>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {layers.map((layer, index) => (
                  <Box key={layer.id} sx={{ position: 'relative' }}>
                    {/* Add Layer Before Button */}
                    {index === 0 && (
                      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                        <LayerInsertMenu
                          onSelectLayer={(layerType) => addLayerBefore(index, layerType)}
                          icon={<AddBeforeIcon />}
                          tooltip="Add layer before"
                        />
                      </Box>
                    )}
                    
                    <Card variant="outlined" sx={{ position: 'relative' }}>
                      <CardContent sx={{ pb: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <span style={{ fontSize: '18px' }}>
                                {LAYER_TYPES.find(lt => lt.type === layer.type)?.icon}
                              </span>
                              {index + 1}. {layer.name}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
                              {Object.entries(layer.params).map(([key, value]) => (
                                <Chip 
                                  key={key} 
                                  label={`${key}: ${value}`} 
                                  size="small"
                                  variant="outlined"
                                />
                              ))}
                              {layer.activation && (
                                <Chip 
                                  label={`activation: ${layer.activation}`} 
                                  size="small"
                                  color="primary"
                                  variant="outlined"
                                />
                              )}
                              {layer.dropout_rate && (
                                <Chip 
                                  label={`dropout: ${layer.dropout_rate}`} 
                                  size="small"
                                  color="warning"
                                  variant="outlined"
                                />
                              )}
                              {layer.batch_norm && (
                                <Chip 
                                  label="batch norm" 
                                  size="small"
                                  color="info"
                                  variant="outlined"
                                />
                              )}
                              {layer.weight_init && (
                                <Chip 
                                  label={`init: ${layer.weight_init}`} 
                                  size="small"
                                  color="secondary"
                                  variant="outlined"
                                />
                              )}
                            </Box>
                          </Box>
                          <Box>
                            <IconButton onClick={() => openLayerConfig(layer)}>
                              <SettingsIcon />
                            </IconButton>
                            <IconButton onClick={() => removeLayer(layer.id)} color="error">
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                    
                    {/* Add Layer After Button */}
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                      <LayerInsertMenu
                        onSelectLayer={(layerType) => addLayerAfter(index, layerType)}
                        icon={<AddAfterIcon />}
                        tooltip="Add layer after"
                      />
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </Paper>
        </Box>

        {/* Network Controls & Summary */}
        <Box>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <ControlsIcon fontSize="small" />
              Network Controls
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<SaveIcon />}
                onClick={saveNetwork}
                disabled={layers.length === 0 || validationErrors.length > 0}
                fullWidth
              >
                Save Network
              </Button>
              <Button
                variant="contained"
                color="success"
                startIcon={<PlayIcon />}
                disabled={layers.length === 0 || validationErrors.length > 0}
                fullWidth
              >
                Train Network
              </Button>
              <Divider />
              <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ChartIcon fontSize="small" />
                Network Summary:
              </Typography>
              <Typography variant="body2">
                Layers: {layers.length}
              </Typography>
              <Typography variant="body2">
                Trainable: {layers.filter(l => ['dense', 'conv2d', 'lstm'].includes(l.type)).length} layers
              </Typography>
              <Typography variant="body2">
                Parameters: ~{estimateParameters(layers).toLocaleString()}
              </Typography>
              <Typography variant="body2" color={validationErrors.length > 0 ? 'error' : 'success'} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                Status: {validationErrors.length > 0 ? (
                  <>
                    <IssuesIcon fontSize="small" />
                    Issues
                  </>
                ) : (
                  <>
                    <ValidIcon fontSize="small" />
                    Valid
                  </>
                )}
              </Typography>
            </Box>
          </Paper>
        </Box>
      </Box>

      {/* Enhanced Layer Configuration Dialog */}
      <EnhancedLayerConfigDialog
        open={configDialogOpen}
        layer={selectedLayer}
        layerIndex={layers.findIndex(l => l.id === selectedLayer?.id)}
        onClose={() => {
          setConfigDialogOpen(false);
          setSelectedLayer(null);
        }}
        onSave={updateLayerConfig}
      />
    </Box>
  );
};

// Layer Insert Menu Component
interface LayerInsertMenuProps {
  onSelectLayer: (layerType: string) => void;
  icon: React.ReactNode;
  tooltip: string;
}

const LayerInsertMenu: React.FC<LayerInsertMenuProps> = ({ onSelectLayer, icon, tooltip }) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleSelectLayer = (layerType: string) => {
    onSelectLayer(layerType);
    handleClose();
  };

  return (
    <>
      <Tooltip title={tooltip}>
        <IconButton
          onClick={handleClick}
          size="small"
          sx={{ 
            bgcolor: 'primary.main', 
            color: 'white',
            '&:hover': { bgcolor: 'primary.dark' }
          }}
        >
          {icon}
        </IconButton>
      </Tooltip>
      <Dialog open={Boolean(anchorEl)} onClose={handleClose}>
        <DialogTitle>Select Layer Type</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 250 }}>
            {LAYER_TYPES.map((layerType) => (
              <Button
                key={layerType.type}
                variant="outlined"
                startIcon={<span>{layerType.icon}</span>}
                onClick={() => handleSelectLayer(layerType.type)}
                sx={{ justifyContent: 'flex-start' }}
              >
                {layerType.name}
              </Button>
            ))}
          </Box>
        </DialogContent>
      </Dialog>
    </>
  );
};

// Helper functions
function getDefaultParams(layerType: string): Record<string, any> {
  switch (layerType) {
    case 'dense':
      return { units: 64 };
    case 'conv2d':
      return { filters: 32, kernel_size: 3, padding: 'same' };
    case 'maxpool2d':
    case 'avgpool2d':
      return { pool_size: 2 };
    case 'flatten':
      return {};
    case 'lstm':
      return { units: 128, return_sequences: false };
    case 'attention':
      return { heads: 8, embed_dim: 64 };
    default:
      return {};
  }
}

function getDefaultHyperparams(layerType: string): Partial<LayerConfig> {
  return {
    weight_init: ['dense', 'conv2d'].includes(layerType) ? 'xavier_uniform' : undefined,
    use_bias: ['dense', 'conv2d'].includes(layerType) ? true : undefined,
  };
}

function estimateParameters(layers: LayerConfig[]): number {
  let totalParams = 0;
  let prevSize = 0;
  
  layers.forEach((layer, index) => {
    if (layer.type === 'dense') {
      const units = layer.params.units || 64;
      const inputSize = index === 0 ? (layer.params.input_size || 784) : prevSize;
      totalParams += (inputSize * units) + (layer.use_bias ? units : 0);
      prevSize = units;
    } else if (layer.type === 'conv2d') {
      const filters = layer.params.filters || 32;
      const kernelSize = layer.params.kernel_size || 3;
      const inputChannels = index === 0 ? 3 : (layers[index-1].params?.filters || 3);
      totalParams += (kernelSize * kernelSize * inputChannels * filters) + (layer.use_bias ? filters : 0);
      prevSize = filters;
    } else if (layer.type === 'lstm') {
      const units = layer.params.units || 128;
      const inputSize = index === 0 ? (layer.params.input_size || 100) : prevSize;
      totalParams += 4 * (inputSize * units + units * units + units); // LSTM gates
      prevSize = units;
    } else if (layer.type === 'maxpool2d' || layer.type === 'avgpool2d') {
      const poolSize = layer.params.pool_size || 2;
      const stride = layer.params.stride || poolSize;
      if (index > 0 && layers[index-1].type === 'conv2d') {
        const inputChannels = layers[index-1].params?.filters || 3;
        prevSize = inputChannels;
      }
    } else if (layer.type === 'flatten') {
      // No parameters for flatten layers
    } 
  });
  
  return totalParams;
}

// Enhanced Layer Configuration Dialog 
interface EnhancedLayerConfigDialogProps {
  open: boolean;
  layer: LayerConfig | null;
  layerIndex: number;
  onClose: () => void;
  onSave: (layer: LayerConfig) => void;
}

const EnhancedLayerConfigDialog: React.FC<EnhancedLayerConfigDialogProps> = ({
  open,
  layer,
  layerIndex,
  onClose,
  onSave,
}) => {
  const [editedLayer, setEditedLayer] = useState<LayerConfig | null>(null);

  React.useEffect(() => {
    if (layer) {
      setEditedLayer({ ...layer });
    }
  }, [layer]);

  const handleSave = () => {
    if (editedLayer) {
      onSave(editedLayer);
    }
  };

  const isFirstLayer = layerIndex === 0;

  if (!editedLayer) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        Configure {editedLayer.name}
        <Typography variant="caption" color="text.secondary" display="block">
          Advanced layer configuration with comprehensive hyperparameters
        </Typography>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {/* Basic Parameters */}
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ParamsIcon fontSize="small" />
                  Basic Parameters
                </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* Input Size for First Layer */}
                {isFirstLayer && ['dense', 'lstm'].includes(editedLayer.type) && (
                  <TextField
                    label="Input Size"
                    type="number"
                    value={editedLayer.params.input_size || ''}
                    onChange={(e) => setEditedLayer({
                      ...editedLayer,
                      params: { ...editedLayer.params, input_size: parseInt(e.target.value) || undefined }
                    })}
                    helperText="Number of input features (required for first layer)"
                    fullWidth
                    required
                  />
                )}

                {/* CNN Input Shape for First Conv Layer */}
                {isFirstLayer && editedLayer.type === 'conv2d' && (
                  <>
                    <TextField
                      label="Input Channels"
                      type="number"
                      value={editedLayer.params.input_channels || ''}
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        params: { ...editedLayer.params, input_channels: parseInt(e.target.value) || 3 }
                      })}
                      helperText="Number of input channels (1 for grayscale, 3 for RGB)"
                      fullWidth
                    />
                    <TextField
                      label="Input Height"
                      type="number"
                      value={editedLayer.params.input_height || ''}
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        params: { ...editedLayer.params, input_height: parseInt(e.target.value) || undefined }
                      })}
                      helperText="Height of input images"
                      fullWidth
                    />
                    <TextField
                      label="Input Width"
                      type="number"
                      value={editedLayer.params.input_width || ''}
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        params: { ...editedLayer.params, input_width: parseInt(e.target.value) || undefined }
                      })}
                      helperText="Width of input images"
                      fullWidth
                    />
                  </>
                )}

                {/* Regular Parameters */}
                {Object.entries(editedLayer.params)
                  .filter(([key]) => !['input_size', 'input_channels', 'input_height', 'input_width', 'padding'].includes(key))
                  .map(([key, value]) => (
                  <TextField
                    key={key}
                    label={key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')}
                    type="number"
                    value={value}
                    onChange={(e) => setEditedLayer({
                      ...editedLayer,
                      params: { ...editedLayer.params, [key]: e.target.value }
                    })}
                    fullWidth
                  />
                ))}

                {/* CNN-specific parameters */}
                {editedLayer.type === 'conv2d' && (
                  <>
                    <TextField
                      label="Stride"
                      type="number"
                      value={editedLayer.stride || 1}
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        stride: parseInt(e.target.value) || 1
                      })}
                      helperText="Stride for convolution (1 = no stride)"
                      fullWidth
                    />
                    <FormControl fullWidth>
                      <InputLabel>Padding</InputLabel>
                      <Select
                        value={editedLayer.padding || 'same'}
                        label="Padding"
                        onChange={(e) => setEditedLayer({
                          ...editedLayer,
                          padding: e.target.value
                        })}
                      >
                        <MenuItem value="same">Same (maintain size)</MenuItem>
                        <MenuItem value="valid">Valid (no padding)</MenuItem>
                        <MenuItem value={0}>None (0)</MenuItem>
                        <MenuItem value={1}>Padding 1</MenuItem>
                        <MenuItem value={2}>Padding 2</MenuItem>
                      </Select>
                    </FormControl>
                  </>
                )}
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Activation & Regularization */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ActivationIcon fontSize="small" />
                  Activation & Regularization
                </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* Activation Function */}
                {['dense', 'conv2d', 'lstm'].includes(editedLayer.type) && (
                  <FormControl fullWidth>
                    <InputLabel>Activation Function</InputLabel>
                    <Select
                      value={editedLayer.activation || 'none'}
                      label="Activation Function"
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        activation: e.target.value === 'none' ? undefined : e.target.value
                      })}
                    >
                      {ACTIVATION_FUNCTIONS.map((activation) => (
                        <MenuItem key={activation} value={activation}>
                          {activation === 'none' ? 'None (Linear) - Best for regression output' : activation.toUpperCase()}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}

                {/* Dropout Rate */}
                <TextField
                  label="Dropout Rate"
                  type="number"
                  inputProps={{ min: 0, max: 1, step: 0.1 }}
                  value={editedLayer.dropout_rate || 0}
                  onChange={(e) => setEditedLayer({
                    ...editedLayer,
                    dropout_rate: parseFloat(e.target.value) || undefined
                  })}
                  helperText="0 = no dropout, 0.5 = 50% dropout"
                  fullWidth
                />

                {/* Batch Normalization */}
                <FormControlLabel
                  control={
                    <Switch
                      checked={editedLayer.batch_norm || false}
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        batch_norm: e.target.checked
                      })}
                    />
                  }
                  label="Enable Batch Normalization"
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Advanced Parameters */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AdvancedIcon fontSize="small" />
                  Advanced Parameters
                </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {/* Learning Rate */}
                <TextField
                  label="Layer Learning Rate (optional)"
                  type="number"
                  inputProps={{ min: 0, step: 0.0001 }}
                  value={editedLayer.learning_rate || ''}
                  onChange={(e) => setEditedLayer({
                    ...editedLayer,
                    learning_rate: e.target.value ? parseFloat(e.target.value) : undefined
                  })}
                  helperText="Leave empty to use global learning rate"
                  fullWidth
                />

                {/* Weight Initialization */}
                {['dense', 'conv2d'].includes(editedLayer.type) && (
                  <FormControl fullWidth>
                    <InputLabel>Weight Initialization</InputLabel>
                    <Select
                      value={editedLayer.weight_init || 'xavier_uniform'}
                      label="Weight Initialization"
                      onChange={(e) => setEditedLayer({
                        ...editedLayer,
                        weight_init: e.target.value
                      })}
                    >
                      {WEIGHT_INITIALIZATIONS.map((init) => (
                        <MenuItem key={init} value={init}>
                          {init.replace('_', ' ').toUpperCase()}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}

                {/* Use Bias */}
                {['dense', 'conv2d'].includes(editedLayer.type) && (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={editedLayer.use_bias !== false}
                        onChange={(e) => setEditedLayer({
                          ...editedLayer,
                          use_bias: e.target.checked
                        })}
                      />
                    }
                    label="Use Bias"
                  />
                )}
              </Box>
            </AccordionDetails>
          </Accordion>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">Save Configuration</Button>
      </DialogActions>
    </Dialog>
  );
};

export default NetworkBuilder; 