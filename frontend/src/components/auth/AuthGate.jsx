import React, { useCallback, useEffect, useRef, useState } from "react";
import { auth, SIGNED_OUT } from "../../api/client.js";
import { IconEye, IconEyeOff } from "../ui/Icons.jsx";
import EarthGlobe from "./EarthGlobe.jsx";

// Sign in / Create account, shown in place of the app until a session exists.
// The real Earth turns on the left (EarthGlobe); the form sits on the right
// with the engine's status under it as a mono readout. Signing in leads with
// what Syntra does; creating an account keeps its own plain title. The first account on a fresh install becomes the admin and can
// only be made on the computer running Syntra (the server enforces both).

function Field({ id, label, error, children, hint }) {
  return (
    <div className={`auth-field ${error ? "has-error" : ""}`}>
      <label htmlFor={id}>{label}</label>
      {children}
      {error ? <p className="auth-field-error" id={`${id}-err`}>{error}</p>
        : hint ? <p className="auth-field-hint mono">{hint}</p> : null}
    </div>
  );
}

// provider marks, one colour each as their brand guidelines ask for sign-in buttons
function GoogleMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.6 5.4 2.6 13.2l7.9 6.2C12.4 13.6 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.1 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.4c-.5 2.9-2.2 5.3-4.6 6.9l7.2 5.6c4.2-3.9 7.1-9.6 7.1-17z" />
      <path fill="#FBBC05" d="M10.5 28.6c-.5-1.4-.8-3-.8-4.6s.3-3.2.8-4.6l-7.9-6.2C1 16.5 0 20.1 0 24s1 7.5 2.6 10.8l7.9-6.2z" />
      <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.2-5.6c-2.2 1.5-5 2.4-8.7 2.4-6.3 0-11.6-4.1-13.5-9.9l-7.9 6.2C6.6 42.6 14.6 48 24 48z" />
    </svg>
  );
}
function GitHubMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.33c-2.23.48-2.7-1.07-2.7-1.07-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.23.83 1.23.83.72 1.22 1.88.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function AuthCard({ status, onSignedIn }) {
  const firstRun = !status.accounts_exist;
  const [mode, setMode] = useState(firstRun ? "signup" : "signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  // a Google / GitHub sign-in that came back with an error puts it here
  const [errors, setErrors] = useState(() => {
    const m = new URLSearchParams(window.location.search).get("signin_error");
    if (m) window.history.replaceState(null, "", window.location.pathname);
    return m ? { form: m } : {};
  });
  const firstInput = useRef(null);
  const providers = Object.entries(status.oauth || {}).filter(([, on]) => on).map(([p]) => p);
  const signup = mode === "signup";

  useEffect(() => { firstInput.current?.focus(); }, [mode]);

  function switchMode(next) {
    setMode(next);
    setErrors({});
    setPassword("");
    setShow(false);
  }

  async function submit(e) {
    e.preventDefault();
    const local = {};
    if (!email.trim()) local.email = "Enter your email";
    if (!password) local.password = "Enter a password";
    else if (signup && password.length < 8) local.password = "Use at least 8 characters";
    setErrors(local);
    if (Object.keys(local).length) return;
    setBusy(true);
    try {
      const r = signup ? await auth.signUp(name, email, password) : await auth.signIn(email, password);
      onSignedIn(r.user);
    } catch (err) {
      const field = err.detail?.field;
      setErrors(field ? { [field]: err.message } : { form: err.message.startsWith("Failed to fetch") ? "The engine is not reachable" : err.message });
      setBusy(false);
    }
  }

  const blockedFirstRun = firstRun && !status.first_run_here;
  const canSwitch = !firstRun && status.signup_open;

  return (
    <form className="auth-card" onSubmit={submit} noValidate>
      <div className="auth-brand">
        <span className="brand-mark" aria-hidden="true" />
        <span className="brand-name">Syntra AI</span>
      </div>

      {firstRun || signup ? (
        <h1 className="auth-title">{firstRun ? "Create the admin account" : "Create account"}</h1>
      ) : (<>
        <h1 className="auth-headline">Every object on the ground, as a layer you can use</h1>
        <p className="auth-lede">Satellite and drone GeoTIFFs in, GeoJSON, KML and Shapefile out</p>
      </>)}
      {firstRun && <p className="auth-lede">The first account manages this installation</p>}

      {blockedFirstRun ? (
        <p className="auth-form-error">Open Syntra on the computer it runs on to create the first account</p>
      ) : (<>
        {signup && (
          <Field id="auth-name" label="Name">
            <input id="auth-name" ref={firstInput} type="text" autoComplete="name" value={name}
                   onChange={(e) => setName(e.target.value)} maxLength={80} />
          </Field>
        )}
        <Field id="auth-email" label="Email" error={errors.email}>
          <input id="auth-email" ref={signup ? undefined : firstInput} type="email" autoComplete={signup ? "email" : "username"}
                 inputMode="email" spellCheck={false} value={email} onChange={(e) => setEmail(e.target.value)}
                 aria-invalid={!!errors.email} aria-describedby={errors.email ? "auth-email-err" : undefined} />
        </Field>
        <Field id="auth-password" label="Password" error={errors.password} hint={signup ? "8 characters or more" : null}>
          <div className="auth-password">
            <input id="auth-password" type={show ? "text" : "password"} value={password}
                   autoComplete={signup ? "new-password" : "current-password"}
                   onChange={(e) => setPassword(e.target.value)}
                   aria-invalid={!!errors.password} aria-describedby={errors.password ? "auth-password-err" : undefined} />
            <button type="button" className="auth-reveal" onClick={() => setShow((s) => !s)}
                    aria-label={show ? "Hide password" : "Show password"} aria-pressed={show}>
              {show ? <IconEyeOff /> : <IconEye />}
            </button>
          </div>
        </Field>

        {errors.form && <p className="auth-form-error" role="alert">{errors.form}</p>}

        <button type="submit" className="auth-submit" disabled={busy}>
          {busy ? (signup ? "Creating…" : "Signing in…") : signup ? "Create account" : "Sign in"}
        </button>
        {providers.length > 0 && (!firstRun || status.first_run_here) && (
          <div className="auth-oauth">
            <span className="auth-or"><span>or</span></span>
            {providers.map((p) => (
              <a key={p} className="auth-provider" href={`/api/auth/oauth/${p}/start`}>
                {p === "google" ? <GoogleMark /> : <GitHubMark />}
                <span>Continue with {p === "google" ? "Google" : "GitHub"}</span>
              </a>
            ))}
          </div>
        )}
      </>)}

      {canSwitch && (
        <p className="auth-switch">
          {signup ? "Have an account?" : "No account yet?"}{" "}
          <button type="button" className="link-button" onClick={() => switchMode(signup ? "signin" : "signup")}>
            {signup ? "Sign in" : "Create one"}
          </button>
        </p>
      )}
      {!firstRun && !status.signup_open && !signup && (
        <p className="auth-switch">New accounts are made by the admin</p>
      )}
    </form>
  );
}

function EngineReadout({ up, onRetry }) {
  return (
    <p className="auth-readout mono" aria-live="polite">
      <span className={`auth-dot ${up ? "up" : up === false ? "down" : ""}`} aria-hidden="true" />
      {up == null ? "Engine · checking" : up ? `Engine · online · ${window.location.host}` : "Engine · not reachable"}
      {up === false && <button type="button" className="link-button" onClick={onRetry}>Retry</button>}
    </p>
  );
}

export default function AuthGate({ children }) {
  const [status, setStatus] = useState(null);       // /api/auth/status
  const [up, setUp] = useState(null);


  const load = useCallback(() => {
    setUp(null);
    auth.status()
      .then((s) => { setStatus(s); setUp(true); })
      .catch(() => { setUp(false); setStatus((cur) => cur || { enabled: true, accounts_exist: true, signup_open: false, user: null, offline: true }); });
  }, []);

  useEffect(load, [load]);

  useEffect(() => {
    const onOut = () => setStatus((s) => (s && s.user ? { ...s, user: null, accounts_exist: true } : s));
    window.addEventListener(SIGNED_OUT, onOut);
    return () => window.removeEventListener(SIGNED_OUT, onOut);
  }, []);

  const signOut = useCallback(async () => {
    await auth.signOut().catch(() => {});
    setStatus((s) => ({ ...s, user: null, accounts_exist: true }));
  }, []);

  if (!status) {
    return <div className="auth-screen"><EngineReadout up={up} onRetry={load} /></div>;
  }
  if (!status.enabled || status.user) {
    return children({ user: status.user, onSignOut: status.enabled ? signOut : null });
  }
  return (
    <main className="auth-screen auth-split">
      <EarthGlobe />
      <div className="auth-stack">
        {status.offline
          ? <div className="auth-card"><div className="auth-brand"><span className="brand-mark" aria-hidden="true" /><span className="brand-name">Syntra AI</span></div>
              <h1 className="auth-title">Sign in</h1>
              <p className="auth-lede">Start the engine, then retry. On Windows run scripts\start-windows.ps1</p></div>
          : <AuthCard key={status.accounts_exist ? "acc" : "first"} status={status}
                      onSignedIn={(user) => setStatus((s) => ({ ...s, user, accounts_exist: true }))} />}
        <EngineReadout up={up} onRetry={load} />
      </div>
    </main>
  );
}
