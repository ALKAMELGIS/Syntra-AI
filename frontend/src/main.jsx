import React from "react";
import ReactDOM from "react-dom/client";
import "leaflet/dist/leaflet.css";
import "./styles.css";
import App from "./App.jsx";
import AuthGate from "./components/AuthGate.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary label="Syntra">
      <AuthGate>{({ user, onSignOut }) => <App user={user} onSignOut={onSignOut} />}</AuthGate>
    </ErrorBoundary>
  </React.StrictMode>
);
