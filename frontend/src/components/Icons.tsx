import React from 'react';
import {
  Psychology as BrainIconMui,
  Storage as DatasetIconMui,
  AccountTree as NetworkIconMui,
  Timeline as VisualizationIconMui,
} from '@mui/icons-material';
import { SvgIconProps } from '@mui/material/SvgIcon';

export const BrainIcon: React.FC<SvgIconProps> = (props) => (
  <BrainIconMui {...props} />
);

export const DatasetIcon: React.FC<SvgIconProps> = (props) => (
  <DatasetIconMui {...props} />
);

export const NetworkIcon: React.FC<SvgIconProps> = (props) => (
  <NetworkIconMui {...props} />
);

export const VisualizationIcon: React.FC<SvgIconProps> = (props) => (
  <VisualizationIconMui {...props} />
); 