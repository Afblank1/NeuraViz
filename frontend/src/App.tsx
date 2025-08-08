import React, { useState } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Paper,
  Tabs,
  Tab
} from '@mui/material';
import { BrainIcon, DatasetIcon, NetworkIcon, VisualizationIcon } from './components/Icons';
import NetworkBuilder from './components/NetworkBuilder';
import DatasetManager from './components/DatasetManager';
import TrainingVisualizer from './components/TrainingVisualizer';
import ModelManager from './components/ModelManager';

// Create Material UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 12,
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      style={{ 
        flex: 1, 
        display: value === index ? 'flex' : 'none',
        flexDirection: 'column',
        overflow: 'auto'
      }}
      {...other}
    >
      {value === index && (
        <Box sx={{ 
          p: 2, 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'auto'
        }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        width: '100vw'
      }}>
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <BrainIcon sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              NeuraViz - Neural Network Visualization & Builder
            </Typography>
          </Toolbar>
        </AppBar>

        <Box sx={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          overflow: 'hidden'
        }}>
          <Paper elevation={1} sx={{ 
            borderRadius: 0,
            flex: 1,
            display: 'flex',
            flexDirection: 'column'
          }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              aria-label="NeuraViz main tabs"
              sx={{ borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}
            >
              <Tab 
                icon={<NetworkIcon />} 
                label="Network Builder" 
                id="tab-0"
                aria-controls="tabpanel-0"
              />
              <Tab 
                icon={<DatasetIcon />} 
                label="Datasets" 
                id="tab-1"
                aria-controls="tabpanel-1"
              />
              <Tab 
                icon={<VisualizationIcon />} 
                label="Training & Visualization" 
                id="tab-2"
                aria-controls="tabpanel-2"
              />
              <Tab 
                icon={<BrainIcon />} 
                label="Model Manager" 
                id="tab-3"
                aria-controls="tabpanel-3"
              />
            </Tabs>

            <TabPanel value={currentTab} index={0}>
              <NetworkBuilder />
            </TabPanel>
            
            <TabPanel value={currentTab} index={1}>
              <DatasetManager />
            </TabPanel>
            
            <TabPanel value={currentTab} index={2}>
              <TrainingVisualizer />
            </TabPanel>
            
            <TabPanel value={currentTab} index={3}>
              <ModelManager />
            </TabPanel>
          </Paper>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
