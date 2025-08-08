import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  AlertTitle,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  AutoFixHigh as AutoIcon,
  Tune as ManualIcon,
  Preview as PreviewIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon,
  PlayArrow as ApplyIcon,
  Refresh as RefreshIcon,
  Transform as DataPreprocessorIcon,
} from '@mui/icons-material';

interface PreprocessingInfo {
  dataset_name: string;
  dataset_type: string;
  feature_analysis: FeatureAnalysis[];
  target_analysis: any;
  recommendations: Recommendation[];
  preprocessing_needed: boolean;
}

interface FeatureAnalysis {
  index: number;
  name: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  range: number;
  has_outliers: boolean;
  is_normalized: boolean;
  is_standardized: boolean;
}

interface Recommendation {
  type: string;
  priority: string;
  message: string;
  suggested_method: string;
}

interface PreprocessingConfig {
  dataset_name: string;
  feature_preprocessing: Record<number, string>;
  target_preprocessing?: string;
  remove_outliers: boolean;
  outlier_method: string;
  custom_ranges?: Record<number, [number, number]>;
}

interface PreviewData {
  feature_name: string;
  scaler_type: string;
  original_stats: any;
  scaled_stats: any;
  sample_original: number[];
  sample_scaled: number[];
}

interface DataPreprocessorProps {
  open: boolean;
  onClose: () => void;
  datasetId: string;
  datasetName: string;
  onPreprocessingComplete: (processedDatasetId: string) => void;
}

const DataPreprocessor: React.FC<DataPreprocessorProps> = ({
  open,
  onClose,
  datasetId,
  datasetName,
  onPreprocessingComplete
}) => {
  const [preprocessingInfo, setPreprocessingInfo] = useState<PreprocessingInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [mode, setMode] = useState<'auto' | 'manual'>('auto');
  const [config, setConfig] = useState<PreprocessingConfig>({
    dataset_name: datasetId,
    feature_preprocessing: {},
    target_preprocessing: undefined,
    remove_outliers: false,
    outlier_method: 'iqr'
  });
  const [previewData, setPreviewData] = useState<Record<number, PreviewData>>({});
  const [selectedTab, setSelectedTab] = useState(0);

  useEffect(() => {
    if (open && datasetId) {
      fetchPreprocessingInfo();
    }
  }, [open, datasetId]);

  const fetchPreprocessingInfo = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}/preprocessing-info`);
      if (response.ok) {
        const info = await response.json();
        setPreprocessingInfo(info);
        
        // Auto configure based on recommendations
        if (mode === 'auto') {
          autoConfigurePreprocessing(info);
        }
      }
    } catch (error) {

    } finally {
      setLoading(false);
    }
  };

  const autoConfigurePreprocessing = (info: PreprocessingInfo) => {
    const autoConfig: PreprocessingConfig = {
      dataset_name: datasetId,
      feature_preprocessing: {},
      remove_outliers: false,
      outlier_method: 'iqr'
    };

    // Apply recommendations automatically
    info.recommendations.forEach(rec => {
      if (rec.type === 'scaling' && rec.priority === 'high') {
        // Apply suggested scaler to all features
        info.feature_analysis.forEach(feature => {
          if (!feature.is_normalized && !feature.is_standardized) {
            autoConfig.feature_preprocessing[feature.index] = rec.suggested_method;
          }
        });
      }
      
      if (rec.type === 'outliers' && rec.priority === 'medium') {
        autoConfig.remove_outliers = true;
        autoConfig.outlier_method = 'iqr';
      }
    });

    // For regression, suggest target scaling if needed
    if (info.dataset_type === 'regression' && info.target_analysis.range > 1000) {
      autoConfig.target_preprocessing = 'StandardScaler';
    }

    setConfig(autoConfig);
  };

  const previewFeaturePreprocessing = async (featureIndex: number, scalerType: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/datasets/${datasetId}/preprocessing-preview?feature_index=${featureIndex}&scaler_type=${scalerType}`
      );
      if (response.ok) {
        const preview = await response.json();
        setPreviewData(prev => ({
          ...prev,
          [featureIndex]: preview
        }));
      }
    } catch (error) {

    }
  };

  const applyPreprocessing = async () => {
    setApplying(true);
    try {
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}/preprocess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        const result = await response.json();
        onPreprocessingComplete(result.processed_dataset_id);
        onClose();
      } else {
        const error = await response.json();
        alert(`Preprocessing failed: ${error.detail}`);
      }
    } catch (error) {
      alert('Error applying preprocessing');
    } finally {
      setApplying(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const formatNumber = (num: number | null | undefined) => {
    if (num === null || num === undefined || isNaN(num)) {
      return 'N/A';
    }
    return num.toFixed(4);
  };

  if (loading) {
    return (
      <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
        <DialogContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <LinearProgress sx={{ mb: 2 }} />
            <Typography>Analyzing dataset for preprocessing recommendations...</Typography>
          </Box>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DataPreprocessorIcon />
          Data Preprocessor - {datasetName}
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {preprocessingInfo && (
          <Box>
            {/* Mode Selection */}
            <Box sx={{ mb: 3 }}>
              <Tabs value={mode} onChange={(_, value) => setMode(value)}>
                <Tab 
                  icon={<AutoIcon />} 
                  label="Auto Mode" 
                  value="auto"
                  iconPosition="start"
                />
                <Tab 
                  icon={<ManualIcon />} 
                  label="Manual Mode" 
                  value="manual"
                  iconPosition="start"
                />
              </Tabs>
            </Box>

            {/* Recommendations */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <InfoIcon fontSize="small" />
                  Analysis & Recommendations
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ mb: 2 }}>
                  <Alert 
                    severity={preprocessingInfo.preprocessing_needed ? 'warning' : 'success'}
                    icon={preprocessingInfo.preprocessing_needed ? <WarningIcon /> : <CheckIcon />}
                  >
                    <AlertTitle>
                      {preprocessingInfo.preprocessing_needed 
                        ? 'Preprocessing Recommended' 
                        : 'Dataset Analysis Complete'
                      }
                    </AlertTitle>
                    {preprocessingInfo.preprocessing_needed 
                      ? 'The analysis found issues that should be addressed before training.'
                      : 'Your dataset looks good! Preprocessing may still improve training performance.'
                    }
                  </Alert>
                </Box>

                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  {preprocessingInfo.recommendations.map((rec, index) => (
                    <Alert 
                      key={index} 
                      severity={getPriorityColor(rec.priority) as any}
                      variant="outlined"
                      sx={{ flex: '1 1 300px', mb: 1 }}
                    >
                      <Typography variant="body2" fontWeight="bold">
                        {rec.type.toUpperCase()} - {rec.priority.toUpperCase()} PRIORITY
                      </Typography>
                      <Typography variant="body2">{rec.message}</Typography>
                      {rec.suggested_method !== 'none' && (
                        <Chip 
                          label={`Suggested: ${rec.suggested_method}`} 
                          size="small" 
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Alert>
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Feature Analysis */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ManualIcon fontSize="small" />
                  Feature Analysis
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Feature</TableCell>
                        <TableCell align="right">Min</TableCell>
                        <TableCell align="right">Max</TableCell>
                        <TableCell align="right">Mean</TableCell>
                        <TableCell align="right">Std</TableCell>
                        <TableCell align="right">Range</TableCell>
                        <TableCell align="center">Status</TableCell>
                        {mode === 'manual' && <TableCell align="center">Preprocessing</TableCell>}
                        <TableCell align="center">Preview</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {preprocessingInfo.feature_analysis.map((feature) => (
                        <TableRow key={feature.index}>
                          <TableCell>{feature.name}</TableCell>
                          <TableCell align="right">{formatNumber(feature.min)}</TableCell>
                          <TableCell align="right">{formatNumber(feature.max)}</TableCell>
                          <TableCell align="right">{formatNumber(feature.mean)}</TableCell>
                          <TableCell align="right">{formatNumber(feature.std)}</TableCell>
                          <TableCell align="right">{formatNumber(feature.range)}</TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', justifyContent: 'center' }}>
                              {feature.is_standardized && <Chip label="Std" size="small" color="success" />}
                              {feature.is_normalized && <Chip label="Norm" size="small" color="info" />}
                              {feature.has_outliers && <Chip label="Outliers" size="small" color="warning" />}
                              {!feature.is_standardized && !feature.is_normalized && <Chip label="Raw" size="small" />}
                            </Box>
                          </TableCell>
                          
                          {mode === 'manual' && (
                            <TableCell align="center">
                              <FormControl size="small" sx={{ minWidth: 120 }}>
                                <Select
                                  value={config.feature_preprocessing[feature.index] || 'none'}
                                  onChange={(e) => setConfig(prev => ({
                                    ...prev,
                                    feature_preprocessing: {
                                      ...prev.feature_preprocessing,
                                      [feature.index]: e.target.value
                                    }
                                  }))}
                                >
                                  <MenuItem value="none">None</MenuItem>
                                  <MenuItem value="StandardScaler">Standard</MenuItem>
                                  <MenuItem value="MinMaxScaler">MinMax</MenuItem>
                                  <MenuItem value="RobustScaler">Robust</MenuItem>
                                </Select>
                              </FormControl>
                            </TableCell>
                          )}
                          
                          <TableCell align="center">
                            <Tooltip title="Preview preprocessing effect">
                              <span>
                                <IconButton
                                  size="small"
                                  onClick={() => {
                                    const scalerType = config.feature_preprocessing[feature.index] || 'StandardScaler';
                                    previewFeaturePreprocessing(feature.index, scalerType);
                                  }}
                                  disabled={!config.feature_preprocessing[feature.index]}
                                >
                                  <PreviewIcon />
                                </IconButton>
                              </span>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </AccordionDetails>
            </Accordion>

            {/* Preview Results */}
            {Object.keys(previewData).length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PreviewIcon fontSize="small" />
                    Preprocessing Preview
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ 
                    display: 'grid', 
                    gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, 
                    gap: 2 
                  }}>
                    {Object.entries(previewData).map(([featureIndex, preview]) => (
                      <Card variant="outlined" key={featureIndex}>
                        <CardContent>
                          <Typography variant="subtitle1" gutterBottom>
                            {preview.feature_name} - {preview.scaler_type}
                          </Typography>
                          
                          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 2 }}>
                            <Box>
                              <Typography variant="subtitle2" color="text.secondary">Original</Typography>
                              <Typography variant="body2">Min: {formatNumber(preview.original_stats.min)}</Typography>
                              <Typography variant="body2">Max: {formatNumber(preview.original_stats.max)}</Typography>
                              <Typography variant="body2">Mean: {formatNumber(preview.original_stats.mean)}</Typography>
                              <Typography variant="body2">Std: {formatNumber(preview.original_stats.std)}</Typography>
                            </Box>
                            <Box>
                              <Typography variant="subtitle2" color="text.secondary">After Scaling</Typography>
                              <Typography variant="body2">Min: {formatNumber(preview.scaled_stats.min)}</Typography>
                              <Typography variant="body2">Max: {formatNumber(preview.scaled_stats.max)}</Typography>
                              <Typography variant="body2">Mean: {formatNumber(preview.scaled_stats.mean)}</Typography>
                              <Typography variant="body2">Std: {formatNumber(preview.scaled_stats.std)}</Typography>
                            </Box>
                          </Box>
                          
                          <Typography variant="caption" color="text.secondary">
                            Sample values: {preview.sample_original.slice(0, 3).map(v => formatNumber(v)).join(', ')}... 
                            → {preview.sample_scaled.slice(0, 3).map(v => formatNumber(v)).join(', ')}...
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Additional Options */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ManualIcon fontSize="small" />
                  Additional Options
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
                  {/* Outlier Removal */}
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1" gutterBottom>Outlier Removal</Typography>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={config.remove_outliers}
                            onChange={(e) => setConfig(prev => ({
                              ...prev,
                              remove_outliers: e.target.checked
                            }))}
                          />
                        }
                        label="Remove outliers from training data"
                      />
                      {config.remove_outliers && (
                        <FormControl fullWidth size="small" sx={{ mt: 2 }}>
                          <InputLabel>Outlier Detection Method</InputLabel>
                          <Select
                            value={config.outlier_method}
                            label="Outlier Detection Method"
                            onChange={(e) => setConfig(prev => ({
                              ...prev,
                              outlier_method: e.target.value
                            }))}
                          >
                            <MenuItem value="iqr">IQR Method</MenuItem>
                            <MenuItem value="zscore">Z-Score Method</MenuItem>
                          </Select>
                        </FormControl>
                      )}
                    </CardContent>
                  </Card>

                  {/* Target Preprocessing regression */}
                  {preprocessingInfo.dataset_type === 'regression' && (
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Target Preprocessing</Typography>
                        <FormControl fullWidth size="small">
                          <InputLabel>Target Scaling</InputLabel>
                          <Select
                            value={config.target_preprocessing || 'none'}
                            label="Target Scaling"
                            onChange={(e) => setConfig(prev => ({
                              ...prev,
                              target_preprocessing: e.target.value === 'none' ? undefined : e.target.value
                            }))}
                          >
                            <MenuItem value="none">None</MenuItem>
                            <MenuItem value="StandardScaler">Standard Scaler</MenuItem>
                            <MenuItem value="MinMaxScaler">MinMax Scaler</MenuItem>
                            <MenuItem value="RobustScaler">Robust Scaler</MenuItem>
                          </Select>
                        </FormControl>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Current range: {formatNumber(preprocessingInfo.target_analysis.min)} to {formatNumber(preprocessingInfo.target_analysis.max)}
                        </Typography>
                      </CardContent>
                    </Card>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          onClick={fetchPreprocessingInfo}
          startIcon={<RefreshIcon />}
        >
          Refresh Analysis
        </Button>
        <Button
          variant="contained"
          onClick={applyPreprocessing}
          startIcon={applying ? <LinearProgress /> : <ApplyIcon />}
          disabled={applying || !preprocessingInfo}
        >
          {applying ? 'Applying...' : 'Apply Preprocessing'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DataPreprocessor; 