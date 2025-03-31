import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {Toaster} from "@/components/ui/sonner.tsx";
import './global.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
      <Toaster position={"top-left"} expand={true} richColors/>
      <App />
  </StrictMode>,
)
