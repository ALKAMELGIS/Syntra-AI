import React from "react";

// A fault in one part of the interface must not blank the whole system: the
// failed part shows what happened and a way back; the map, the session and
// the jobs on the server are untouched.
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[Syntra] interface error", error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="error-boundary" role="alert">
        <p className="error-boundary-title">{this.props.label || "This part of Syntra"} stopped with an error</p>
        <p className="error-boundary-detail mono">{String(this.state.error?.message || this.state.error)}</p>
        <button type="button" className="quiet-button" onClick={() => window.location.reload()}>Reload</button>
      </div>
    );
  }
}
