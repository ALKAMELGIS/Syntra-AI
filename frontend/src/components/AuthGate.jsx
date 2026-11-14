import React, { useCallback, useEffect, useRef, useState } from "react";
import { auth, SIGNED_OUT } from "../api/client.js";
import { IconEye, IconEyeOff } from "./Icons.jsx";

// Sign in / Create account, shown in place of the app until a session exists.
// One card on the survey-grid ground; the engine's status sits under it as a
// mono readout. The first account on a fresh install becomes the admin and can
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

function AuthCard({ status, onSignedIn }) {
  const firstRun = !status.accounts_exist;
  const [mode, setMode] = useState(firstRun ? "signup" : "signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState({});
  const firstInput = useRef(null);
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

      <h1 className="auth-title">
        {firstRun ? "Create the admin account" : signup ? "Create account" : "Sign in"}
      </h1>
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

  useEffect(() => {
    const theme = localStorage.getItem("syntra_theme") || "dark";
    document.documentElement.classList.toggle("theme-light", theme === "light");
  }, []);

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
    <main className="auth-screen">
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
