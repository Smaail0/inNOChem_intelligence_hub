import { Component, type ErrorInfo, type ReactNode } from "react";

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("InNOChem frontend render error", error, info);
  }

  render() {
    if (!this.state.error) {
      return this.props.children;
    }

    return (
      <main className="app-shell">
        <section className="hero-panel">
          <div className="eyebrow">Frontend recovery</div>
          <h1>Something blocked the interface</h1>
          <p>
            The app caught a frontend runtime error instead of leaving a blank page.
            Refresh after saving the latest code, or check the browser console for details.
          </p>
          <div className="runtime-error-card">
            <strong>{this.state.error.name}</strong>
            <pre>{this.state.error.message}</pre>
          </div>
        </section>
      </main>
    );
  }
}

