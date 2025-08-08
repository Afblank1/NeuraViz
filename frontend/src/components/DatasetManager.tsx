import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  LinearProgress,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Upload as UploadIcon,
  Visibility as ViewIcon,
  Info as InfoIcon,
  ExpandMore as ExpandMoreIcon,
  DataObject as DataIcon,
  Image as ImageIcon,
  Psychology as MLIcon,
  TextFields as TextIcon,
  Science as SyntheticIcon,
  LocalHospital as MedicalIcon,
  CleaningServices as PreprocessIcon
} from '@mui/icons-material';
import DataPreprocessor from './DataPreprocessor';

interface Dataset {
  name: string;
  type: 'regression' | 'classification';
  category: string;
  input_size: number;
  output_size: number;
  description: string;
  sample_count: number;
  features: string[];
  classes: string[];
  X_train: number[][];
  X_test: number[][];
  y_train: number[];
  y_test: number[];
  preprocessing?: any;
  original_dataset?: string;
}

interface DatasetCategories {
  [category: string]: Dataset[];
}

const CATEGORY_INFO = {
  classic: {
    name: 'Classic ML Datasets',
    icon: <MLIcon />,
    color: '#2196f3',
    description: 'Well-known benchmark datasets from machine learning literature'
  },
  synthetic: {
    name: 'Synthetic Datasets',
    icon: <SyntheticIcon />,
    color: '#4caf50',
    description: 'Artificially generated datasets for learning and experimentation'
  },
  image: {
    name: 'Image Datasets',
    icon: <ImageIcon />,
    color: '#ff9800',
    description: 'Computer vision datasets for image classification tasks'
  },
  text: {
    name: 'Text Datasets',
    icon: <TextIcon />,
    color: '#9c27b0',
    description: 'Natural language processing datasets for text analysis'
  },
  medical: {
    name: 'Medical Datasets',
    icon: <MedicalIcon />,
    color: '#f44336',
    description: 'Healthcare and biomedical datasets'
  },
  custom: {
    name: 'Custom Datasets',
    icon: <DataIcon />,
    color: '#607d8b',
    description: 'User-uploaded custom datasets'
  },
  preprocessed: {
    name: 'Preprocessed Datasets',
    icon: <PreprocessIcon />,
    color: '#795548',
    description: 'Datasets that have been preprocessed with scaling and outlier removal'
  }
};

const DatasetManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Record<string, Dataset>>({});
  const [datasetCategories, setDatasetCategories] = useState<DatasetCategories>({});
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);
  
  // Preprocessing states
  const [preprocessorOpen, setPreprocessorOpen] = useState(false);
  const [selectedDatasetForPreprocessing, setSelectedDatasetForPreprocessing] = useState<{id: string, name: string} | null>(null);

  useEffect(() => {
    fetchDatasets();
    fetchDatasetCategories();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await fetch('http://localhost:8000/datasets');
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
      }
    } catch (error) {

    } finally {
      setLoading(false);
    }
  };

  const fetchDatasetCategories = async () => {
    try {
      const response = await fetch('http://localhost:8000/datasets/categories');
      if (response.ok) {
        const data = await response.json();
        setDatasetCategories(data);
      }
    } catch (error) {

    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload-dataset', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Dataset uploaded successfully! ID: ${result.dataset_id}`);
        fetchDatasets();
        fetchDatasetCategories();
      } else {
        alert('Failed to upload dataset');
      }
    } catch (error) {
      alert('Error uploading dataset');
    } finally {
      setUploading(false);
    }
  };

  const openDatasetDetails = async (datasetId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/datasets/${datasetId}`);
      if (response.ok) {
        const dataset = await response.json();
        setSelectedDataset(dataset);
        setDetailsOpen(true);
      }
    } catch (error) {
      alert(`Error fetching dataset details ${error}`);
    }
  };

  const openPreprocessor = (datasetId: string, datasetName: string) => {
    setSelectedDatasetForPreprocessing({ id: datasetId, name: datasetName });
    setPreprocessorOpen(true);
  };

  const handlePreprocessingComplete = (processedDatasetId: string) => {
    // Refresh datasets 
    fetchDatasets();
    fetchDatasetCategories();
    
    // Close preprocessor
    setPreprocessorOpen(false);
    setSelectedDatasetForPreprocessing(null);
    
    // Show success message
    alert(`Preprocessing completed! New dataset created: ${processedDatasetId}`);
  };

  const getDatasetsByCategory = () => {
    if (selectedCategory === 'all') {
      return Object.entries(datasets);
    }
    return Object.entries(datasets).filter(([_, dataset]) => dataset.category === selectedCategory);
  };

  const getCategoryTabs = () => {
    const categories = ['all', ...Object.keys(datasetCategories)];
    return categories.map(category => {
      const count = category === 'all' 
        ? Object.keys(datasets).length 
        : datasetCategories[category]?.length || 0;
      
      return {
        value: category,
        label: category === 'all' ? 'All Datasets' : CATEGORY_INFO[category]?.name || category,
        count,
        icon: category === 'all' ? <DataIcon /> : CATEGORY_INFO[category]?.icon
      };
    });
  };

  if (loading) {
    return (
      <Box>
        <Typography variant="h4" gutterBottom>Dataset Manager</Typography>
        <LinearProgress />
        <Typography variant="body2" sx={{ mt: 1 }}>Loading dataset collection...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DataIcon />
          Dataset Manager
        </Typography>
        <Box>
          <input
            accept=".csv"
            style={{ display: 'none' }}
            id="dataset-upload"
            type="file"
            onChange={handleFileUpload}
            disabled={uploading}
          />
          <label htmlFor="dataset-upload">
            <Button
              variant="contained"
              component="span"
              startIcon={uploading ? <CircularProgress size={20} /> : <UploadIcon />}
              disabled={uploading}
            >
              {uploading ? 'Uploading...' : 'Upload Dataset'}
            </Button>
          </label>
        </Box>
      </Box>

      {/* Category Navigation */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={selectedCategory}
          onChange={(_, newValue) => setSelectedCategory(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          {getCategoryTabs().map((tab) => (
            <Tab
              key={tab.value}
              value={tab.value}
              icon={
                <Badge badgeContent={tab.count} color="primary">
                  {tab.icon}
                </Badge>
              }
              label={tab.label}
              iconPosition="start"
            />
          ))}
        </Tabs>
      </Paper>

      {/* Category Description */}
      {selectedCategory !== 'all' && CATEGORY_INFO[selectedCategory] && (
        <Alert 
          severity="info" 
          sx={{ mb: 3 }}
          icon={CATEGORY_INFO[selectedCategory].icon}
        >
          <Typography variant="body1" fontWeight="bold">
            {CATEGORY_INFO[selectedCategory].name}
          </Typography>
          <Typography variant="body2">
            {CATEGORY_INFO[selectedCategory].description}
          </Typography>
        </Alert>
      )}

      {/* Dataset Grid */}
      <Box sx={{ 
        display: 'grid', 
        gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }, 
        gap: 3 
      }}>
        {getDatasetsByCategory().map(([datasetId, dataset]) => {
          const categoryInfo = CATEGORY_INFO[dataset.category];
          
          return (
            <Card key={datasetId} variant="outlined" sx={{ height: 'fit-content' }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {categoryInfo?.icon && (
                    <Box sx={{ mr: 1, color: categoryInfo.color }}>
                      {categoryInfo.icon}
                    </Box>
                  )}
                  <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    {dataset.name}
                  </Typography>
                  <Tooltip title="View Details">
                    <IconButton 
                      size="small" 
                      onClick={() => openDatasetDetails(datasetId)}
                    >
                      <ViewIcon />
                    </IconButton>
                  </Tooltip>
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {dataset.description}
                </Typography>

                <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                  <Chip 
                    label={dataset.type} 
                    size="small"
                    color={dataset.type === 'classification' ? 'primary' : 'success'}
                    variant="outlined"
                  />
                  <Chip 
                    label={`${dataset.sample_count} samples`} 
                    size="small"
                    variant="outlined"
                  />
                  <Chip 
                    label={`${dataset.input_size} features`} 
                    size="small"
                    variant="outlined"
                  />
                  {dataset.type === 'classification' && (
                    <Chip 
                      label={`${dataset.output_size} classes`} 
                      size="small"
                      variant="outlined"
                    />
                  )}
                  {dataset.preprocessing && (
                    <Chip 
                      label="Preprocessed" 
                      size="small"
                      color="secondary"
                      icon={<PreprocessIcon />}
                    />
                  )}
                  {dataset.original_dataset && (
                    <Chip 
                      label={`From: ${dataset.original_dataset}`} 
                      size="small"
                      variant="outlined"
                      color="info"
                    />
                  )}
                </Box>

                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    fullWidth
                    startIcon={<InfoIcon />}
                    onClick={() => openDatasetDetails(datasetId)}
                  >
                    View Details
                  </Button>
                  {dataset.category !== 'preprocessed' && (
                    <Tooltip title="Preprocess this dataset">
                      <span>
                        <Button
                          variant="outlined"
                          color="secondary"
                          startIcon={<PreprocessIcon />}
                          onClick={() => openPreprocessor(datasetId, dataset.name)}
                          sx={{ minWidth: 'auto', px: 2 }}
                        >
                          Prep
                        </Button>
                      </span>
                    </Tooltip>
                  )}
                </Box>
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {Object.keys(datasets).length === 0 && (
        <Alert severity="info">
          No datasets available. Upload a CSV file to get started!
        </Alert>
      )}

      {/* Dataset Details Dialog */}
      <Dialog 
        open={detailsOpen} 
        onClose={() => setDetailsOpen(false)} 
        maxWidth="lg" 
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {selectedDataset && CATEGORY_INFO[selectedDataset.category]?.icon}
            {selectedDataset?.name} - Dataset Details
          </Box>
        </DialogTitle>
        <DialogContent>
          {selectedDataset && (
            <Box>
              {/* Dataset Overview */}
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <InfoIcon fontSize="small" />
                    Dataset Overview
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ 
                    display: 'grid', 
                    gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, 
                    gap: 2 
                  }}>
                    <Box>
                      <Typography variant="subtitle2">Description:</Typography>
                      <Typography>{selectedDataset.description}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Category:</Typography>
                      <Typography>{CATEGORY_INFO[selectedDataset.category]?.name || selectedDataset.category}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Type:</Typography>
                      <Chip 
                        label={selectedDataset.type} 
                        size="small"
                        color={selectedDataset.type === 'classification' ? 'primary' : 'success'}
                      />
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Total Samples:</Typography>
                      <Typography>{selectedDataset.sample_count}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Input Features:</Typography>
                      <Typography>{selectedDataset.input_size}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Output Size:</Typography>
                      <Typography>{selectedDataset.output_size}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Training Samples:</Typography>
                      <Typography>{selectedDataset.X_train?.length || 0}</Typography>
                    </Box>
                    <Box>
                      <Typography variant="subtitle2">Test Samples:</Typography>
                      <Typography>{selectedDataset.X_test?.length || 0}</Typography>
                    </Box>
                  </Box>
                </AccordionDetails>
              </Accordion>

              {/* Features & Classes */}
              {((selectedDataset.features?.length || 0) > 0 || (selectedDataset.classes?.length || 0) > 0) && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <TextIcon fontSize="small" />
                      Features & Labels
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    {(selectedDataset.features?.length || 0) > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>Features:</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {(selectedDataset.features || []).map((feature, index) => (
                            <Chip key={index} label={feature} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                    
                    {(selectedDataset.classes?.length || 0) > 0 && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom>Classes:</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {(selectedDataset.classes || []).map((className, index) => (
                            <Chip key={index} label={className} size="small" color="primary" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Preprocessing Info */}
              {selectedDataset.preprocessing && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">
                      <PreprocessIcon />
                      Preprocessing Applied
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ mb: 2 }}>
                      <Alert severity="info" icon={<PreprocessIcon />}>
                        <Typography variant="body2">
                          This dataset has been preprocessed. Original dataset: {selectedDataset.original_dataset || 'Unknown'}
                        </Typography>
                      </Alert>
                    </Box>
                    
                    {selectedDataset.preprocessing.applied_transformations && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>Applied Transformations:</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {selectedDataset.preprocessing.applied_transformations.map((transform: string, index: number) => (
                            <Chip key={index} label={transform} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                    
                    {selectedDataset.preprocessing.removed_samples > 0 && (
                      <Typography variant="body2" color="text.secondary">
                        Removed {selectedDataset.preprocessing.removed_samples} outlier samples during preprocessing.
                      </Typography>
                    )}
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Data Preview */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ViewIcon fontSize="small" />
                    Data Preview
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          {(selectedDataset.features?.length || 0) > 0 ? (
                            (selectedDataset.features || []).map((feature, index) => (
                              <TableCell key={index}>{feature}</TableCell>
                            ))
                          ) : (
                            Array.from({ length: selectedDataset.input_size || 0 }, (_, i) => (
                              <TableCell key={i}>Feature {i + 1}</TableCell>
                            ))
                          )}
                          <TableCell><strong>Target</strong></TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(selectedDataset.X_train || []).slice(0, 10).map((sample, index) => (
                          <TableRow key={index}>
                            {sample.map((value, featureIndex) => (
                              <TableCell key={featureIndex}>
                                {typeof value === 'number' ? value.toFixed(3) : value}
                              </TableCell>
                            ))}
                            <TableCell>
                              <strong>
                                {(selectedDataset.classes?.length || 0) > 0 && selectedDataset.y_train?.[index] < (selectedDataset.classes?.length || 0)
                                  ? selectedDataset.classes[selectedDataset.y_train[index]]
                                  : selectedDataset.y_train?.[index] || 'N/A'
                                }
                              </strong>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                    Showing first 10 samples of training data
                  </Typography>
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Data Preprocessor Dialog */}
      {selectedDatasetForPreprocessing && (
        <DataPreprocessor
          open={preprocessorOpen}
          onClose={() => {
            setPreprocessorOpen(false);
            setSelectedDatasetForPreprocessing(null);
          }}
          datasetId={selectedDatasetForPreprocessing.id}
          datasetName={selectedDatasetForPreprocessing.name}
          onPreprocessingComplete={handlePreprocessingComplete}
        />
      )}
    </Box>
  );
};

export default DatasetManager; 