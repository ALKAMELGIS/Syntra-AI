// Entry point: applies the saved appearance, then renders the app behind the sign-in gate.
import React from "react";
import ReactDOM from "react-dom/client";
import "leaflet/dist/leaflet.css";
import "./styles/tokens.css";
import "./styles/app.css";
import App from "./App.jsx";
import AuthGate from "./components/auth/AuthGate.jsx";
import ErrorBoundary from "./components/ui/ErrorBoundary.jsx";
import { applyAppearance, watchSystemTheme } from "./lib/prefs.js";

applyAppearance();
watchSystemTheme();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary label="Syntra">
      <AuthGate>{({ user, onSignOut }) => <App user={user} onSignOut={onSignOut} />}</AuthGate>
    </ErrorBoundary>
  </React.StrictMode>
);
