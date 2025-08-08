/// <reference types="react-scripts" />

// Temporary fix for Material UI Grid TypeScript issues
declare module '@mui/material/Grid' {
  interface GridProps {
    item?: boolean;
    container?: boolean;
    xs?: boolean | number;
    sm?: boolean | number;
    md?: boolean | number;
    lg?: boolean | number;
    xl?: boolean | number;
    spacing?: number;
  }
}
