// The owner's console inside Settings: Users (balances, runs, add credits,
// disable), Prices (free credits, Run cost by area, packages), Payments (Stripe
// keys, payments) and Sign-in (open sign-up, Google and GitHub apps).
import React, { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client.js";

const n = (v) => Number(v || 0).toLocaleString("en-US");
const ago = (t) => {
  if (!t) return "—";
  const s = Date.now() / 1000 - (typeof t === "string" ? Date.parse(t) / 1000 : t);
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min`;
  if (s < 86400) return `${Math.round(s / 3600)} h`;
  return `${Math.round(s / 86400)} d`;
};
const usd = (cents) => `$${(cents / 100).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;

function Group({ title, children, aside }) {
  return (
    <section className="set-group">
      <div className="adm-group-head"><h3 className="set-group-title">{title}</h3>{aside}</div>
      <div className="set-card">{children}</div>
    </section>
  );
}

function Stat({ label, value }) {
  return <div className="adm-stat"><span className="adm-stat-v mono">{value}</span><span className="adm-stat-l">{label}</span></div>;
}

// --- Users -------------------------------------------------------------------
function UserRow({ u, onChanged }) {
  const [open, setOpen] = useState(false);
  const [amount, setAmount] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState(null);
  const [hist, setHist] = useState(null);
  const owner = u.role === "owner";
  async function act(fn) {
    setErr(null);
    try { await fn(); onChanged(); } catch (e) { setErr(e.message); }
  }
  useEffect(() => { if (open && !owner) api.adminLedger(u.id).then(setHist).catch(() => setHist([])); }, [open, owner, u.id, u.credits]);
  return (
    <>
      <tr className={`adm-row ${open ? "on" : ""} ${u.status !== "active" ? "off" : ""}`} onClick={() => setOpen((o) => !o)}>
        <td><span className="adm-name">{u.name}</span><span className="adm-email mono">{u.email}</span></td>
        <td>{owner ? "Owner" : u.status === "active" ? "Member" : "Disabled"}{u.provider !== "password" && <span className="adm-tag">{u.provider}</span>}</td>
        <td className="mono num">{owner ? "∞" : n(u.credits)}</td>
        <td className="mono num">{n(u.runs)}</td>
        <td className="mono num">{ago(u.last_seen)}</td>
      </tr>
      {open && (
        <tr className="adm-detail"><td colSpan={5}>
          {owner ? <p className="rs-dim">The owner's runs are free and have no limit</p> : (<>
            <div className="adm-actions">
              <input className="train-input mono" type="number" placeholder="Credits" value={amount} onChange={(e) => setAmount(e.target.value)} aria-label="Credits" />
              <input className="train-input" placeholder="Note" value={note} onChange={(e) => setNote(e.target.value)} aria-label="Note" maxLength={120} />
              <button type="button" className="quiet-button" disabled={!amount} onClick={() => act(() => api.adminCredits(u.id, { delta: Number(amount), note }))}>Add</button>
              <button type="button" className="quiet-button" disabled={!amount} onClick={() => act(() => api.adminCredits(u.id, { set_to: Number(amount), note }))}>Set</button>
              <span className="adm-spacer" />
              <button type="button" className="quiet-button" onClick={() => act(() => api.adminPatchUser(u.id, { status: u.status === "active" ? "disabled" : "active" }))}>
                {u.status === "active" ? "Disable" : "Enable"}
              </button>
              <button type="button" className="quiet-button danger" onClick={() => { if (window.confirm(`Delete ${u.email}? Their images and results stay, owned by you`)) act(() => api.adminDeleteUser(u.id)); }}>Delete</button>
            </div>
            <p className="rs-dim mono">Joined {new Date(u.created_at * 1000).toLocaleDateString("en-GB")} · spent {n(u.spent_30d)} in 30 d · last run {ago(u.last_run)}</p>
            {hist && hist.length > 0 && (
              <table className="cr-table adm-hist"><tbody>
                {hist.slice(0, 6).map((h) => (
                  <tr key={h.id}><td className="mono cr-date">{new Date(h.created_at * 1000).toLocaleDateString("en-GB", { day: "numeric", month: "short" })}</td>
                    <td>{h.reason}</td><td className={`mono ${h.delta > 0 ? "cr-plus" : ""}`}>{h.delta > 0 ? "+" : ""}{n(h.delta)}</td><td className="mono rs-dim">{n(h.balance)}</td></tr>
                ))}
              </tbody></table>
            )}
          </>)}
          {err && <p className="lp-error">{err}</p>}
        </td></tr>
      )}
    </>
  );
}

export function Users() {
  const [rows, setRows] = useState(null);
  const [sum, setSum] = useState(null);
  const [q, setQ] = useState("");
  const [err, setErr] = useState(null);
  const load = () => {
    api.adminUsers().then(setRows).catch((e) => setErr(e.message));
    api.adminSummary().then(setSum).catch(() => {});
  };
  useEffect(load, []);
  const shown = useMemo(() => (rows || []).filter((u) => !q || `${u.name} ${u.email}`.toLowerCase().includes(q.toLowerCase())), [rows, q]);
  return (
    <>
      {sum && (
        <div className="adm-stats">
          <Stat label="Accounts" value={n(sum.users)} />
          <Stat label="Active 7 d" value={n(sum.active_7d)} />
          <Stat label="Runs" value={n(sum.runs)} />
          <Stat label="Credits held" value={n(sum.credits_held)} />
          <Stat label="Revenue 30 d" value={usd(sum.revenue_cents_30d)} />
        </div>
      )}
      <Group title="Accounts" aside={<input className="train-input adm-search" placeholder="Search" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search accounts" />}>
        {err ? <p className="lp-error set-pad">{err}</p> : !rows ? <p className="rs-dim set-pad">…</p> : (
          <table className="adm-table">
            <thead><tr><th>Account</th><th>Role</th><th className="num">Credits</th><th className="num">Runs</th><th className="num">Seen</th></tr></thead>
            <tbody>{shown.map((u) => <UserRow key={u.id} u={u} onChanged={load} />)}</tbody>
          </table>
        )}
      </Group>
    </>
  );
}

// --- Prices ------------------------------------------------------------------
export function Prices() {
  const [p, setP] = useState(null);
  const [saved, setSaved] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api.adminPricing().then(setP).catch((e) => setErr(e.message)); }, []);
  if (!p) return <p className="rs-dim">{err || "…"}</p>;
  const set = (k, v) => { setSaved(null); setP({ ...p, [k]: v }); };
  const tier = (i, k, v) => set("run_tiers", p.run_tiers.map((t, j) => (j === i ? { ...t, [k]: Number(v) } : t)));
  const pack = (i, k, v) => set("packages", p.packages.map((t, j) => (j === i ? { ...t, [k]: k === "id" ? v : Number(v) } : t)));
  async function save() {
    setErr(null);
    try { const { defaults, ...rest } = p; setP({ ...(await api.adminSavePricing(rest)), defaults }); setSaved(true); }    
    catch (e) { setErr(e.message); }
  }
  return (
    <>
      <Group title="New accounts">
        <div className="set-row"><div className="set-row-text"><span className="set-row-label">Free credits</span><span className="set-row-hint">Given once when an account is made</span></div>
          <input className="train-input mono adm-num" type="number" value={p.free_credits} onChange={(e) => set("free_credits", Number(e.target.value))} aria-label="Free credits" /></div>
      </Group>
      <Group title="A Run costs">
        {p.run_tiers.map((t, i) => (
          <div className="set-row" key={i}>
            <div className="set-row-text"><span className="set-row-label">Up to</span></div>
            <input className="train-input mono adm-num" type="number" value={t.up_to_km2} onChange={(e) => tier(i, "up_to_km2", e.target.value)} aria-label="Area limit km²" /><span className="rs-dim">km²</span>
            <input className="train-input mono adm-num" type="number" value={t.credits} onChange={(e) => tier(i, "credits", e.target.value)} aria-label="Credits" /><span className="rs-dim">credits</span>
            <button type="button" className="lp-icon" aria-label="Remove tier" disabled={p.run_tiers.length < 2} onClick={() => set("run_tiers", p.run_tiers.filter((_, j) => j !== i))}>×</button>
          </div>
        ))}
        <div className="set-row">
          <div className="set-row-text"><span className="set-row-label">Beyond the last</span></div>
          <span className="rs-dim">+</span>
          <input className="train-input mono adm-num" type="number" value={p.over_last.credits} onChange={(e) => set("over_last", { ...p.over_last, credits: Number(e.target.value) })} aria-label="Credits per step" />
          <span className="rs-dim">per</span>
          <input className="train-input mono adm-num" type="number" value={p.over_last.per_km2} onChange={(e) => set("over_last", { ...p.over_last, per_km2: Number(e.target.value) })} aria-label="Step km²" /><span className="rs-dim">km²</span>
        </div>
        <div className="set-pad"><button type="button" className="link-button" onClick={() => { const l = p.run_tiers[p.run_tiers.length - 1]; set("run_tiers", [...p.run_tiers, { up_to_km2: l.up_to_km2 * 2, credits: l.credits + 50 }]); }}>Add tier</button></div>
      </Group>
      <Group title="Packages">
        {p.packages.map((k, i) => (
          <div className="set-row" key={i}>
            <span className="rs-dim">$</span>
            <input className="train-input mono adm-num" type="number" value={k.usd} onChange={(e) => pack(i, "usd", e.target.value)} aria-label="Price USD" />
            <input className="train-input mono adm-num" type="number" value={k.credits} onChange={(e) => pack(i, "credits", e.target.value)} aria-label="Credits" /><span className="rs-dim">credits</span>
            <span className="mono rs-dim adm-rate">{k.usd ? n(Math.round(k.credits / k.usd)) : "–"} / $</span>
            <button type="button" className="lp-icon" aria-label="Remove package" disabled={p.packages.length < 2} onClick={() => set("packages", p.packages.filter((_, j) => j !== i))}>×</button>
          </div>
        ))}
        <div className="set-pad"><button type="button" className="link-button" onClick={() => set("packages", [...p.packages, { id: `p${Date.now() % 100000}`, usd: 200, credits: 32000 }])}>Add package</button></div>
      </Group>
      <div className="adm-save">
        {err && <span className="lp-error">{err}</span>}
        {saved && <span className="rs-dim">Saved</span>}
        <button type="button" className="rs-primary adm-save-btn" onClick={save}>Save prices</button>
      </div>
    </>
  );
}

// --- Payments ---------------------------------------------------------------
export function Payments() {
  const [st, setSt] = useState(null);
  const [key, setKey] = useState("");
  const [hook, setHook] = useState("");
  const [list, setList] = useState([]);
  const [err, setErr] = useState(null);
  useEffect(() => { api.adminStripe().then(setSt).catch((e) => setErr(e.message)); api.adminPayments().then(setList).catch(() => {}); }, []);
  async function save(body) {
    setErr(null);
    try { setSt(await api.adminSaveStripe(body)); setKey(""); setHook(""); } catch (e) { setErr(e.message); }
  }
  return (
    <>
      <Group title="Stripe">
        <div className="set-row"><div className="set-row-text"><span className="set-row-label">Secret key</span><span className="set-row-hint">{st?.secret_key ? "Saved" : "sk_live_… or sk_test_… from the Stripe dashboard"}</span></div>
          <input className="train-input mono adm-key" type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)} placeholder={st?.secret_key ? "••••••••" : "sk_…"} aria-label="Stripe secret key" />
          <button type="button" className="quiet-button" disabled={!key} onClick={() => save({ secret_key: key })}>Save</button>
          {st?.secret_key && <button type="button" className="quiet-button" onClick={() => save({ secret_key: "" })}>Remove</button>}
        </div>
        <div className="set-row"><div className="set-row-text"><span className="set-row-label">Webhook secret</span><span className="set-row-hint">{st?.webhook_secret ? "Saved" : "Only for a public server: whsec_…"}</span></div>
          <input className="train-input mono adm-key" type="password" autoComplete="off" value={hook} onChange={(e) => setHook(e.target.value)} placeholder={st?.webhook_secret ? "••••••••" : "whsec_…"} aria-label="Webhook signing secret" />
          <button type="button" className="quiet-button" disabled={!hook} onClick={() => save({ webhook_secret: hook })}>Save</button>
        </div>
        {err && <p className="lp-error set-pad">{err}</p>}
      </Group>
      <Group title="Payments">
        {!list.length ? <p className="rs-dim set-pad">No payments yet</p> : (
          <table className="adm-table"><thead><tr><th>Date</th><th>Account</th><th className="num">Credits</th><th className="num">Amount</th><th>Status</th></tr></thead>
            <tbody>{list.map((x) => (
              <tr key={x.id}><td className="mono">{new Date(x.created_at * 1000).toLocaleDateString("en-GB")}</td><td className="mono">#{x.user_id}</td>
                <td className="mono num">{n(x.credits)}</td><td className="mono num">{usd(x.amount_cents)}</td><td>{x.status === "paid" ? "Paid" : "Open"}</td></tr>
            ))}</tbody></table>
        )}
      </Group>
    </>
  );
}

// --- Sign-in ----------------------------------------------------------------
function Provider({ id, label, st, onSave }) {
  const [cid, setCid] = useState(st.client_id || "");
  const [sec, setSec] = useState("");
  const cb = `${st.callback_base}${id}/callback`;
  return (
    <Group title={label}>
      <div className="set-row"><div className="set-row-text"><span className="set-row-label">Callback URL</span><span className="set-row-hint">Paste into the {label} OAuth app</span></div>
        <code className="mono adm-cb" title={cb}>{cb}</code>
        <button type="button" className="quiet-button" onClick={() => navigator.clipboard?.writeText(cb)}>Copy</button></div>
      <div className="set-row"><div className="set-row-text"><span className="set-row-label">Client ID</span></div>
        <input className="train-input mono adm-key" value={cid} onChange={(e) => setCid(e.target.value)} aria-label={`${label} client ID`} /></div>
      <div className="set-row"><div className="set-row-text"><span className="set-row-label">Client secret</span><span className="set-row-hint">{st.secret ? "Saved" : "Not set"}</span></div>
        <input className="train-input mono adm-key" type="password" autoComplete="off" value={sec} onChange={(e) => setSec(e.target.value)} placeholder={st.secret ? "••••••••" : ""} aria-label={`${label} client secret`} />
        <button type="button" className="quiet-button" onClick={() => { onSave({ provider: id, client_id: cid, ...(sec ? { client_secret: sec } : {}) }); setSec(""); }}>Save</button></div>
    </Group>
  );
}

export function SignIn() {
  const [st, setSt] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api.adminSignin().then(setSt).catch((e) => setErr(e.message)); }, []);
  async function save(b) { setErr(null); try { setSt(await api.adminSaveSignin(b)); } catch (e) { setErr(e.message); } }
  if (!st) return <p className="rs-dim">{err || "…"}</p>;
  return (
    <>
      <Group title="Accounts">
        <div className="set-row"><div className="set-row-text"><span className="set-row-label">New accounts</span><span className="set-row-hint">Closed: only accounts you already have can sign in</span></div>
          <div className="set-seg" role="radiogroup" aria-label="New accounts">
            {[[true, "Open"], [false, "Closed"]].map(([v, t]) => (
              <button key={t} type="button" role="radio" aria-checked={st.signup_open === v} className={st.signup_open === v ? "on" : ""} onClick={() => save({ signup_open: v })}>{t}</button>
            ))}
          </div></div>
      </Group>
      <Provider id="google" label="Google" st={{ ...st.providers.google, callback_base: st.callback_base }} onSave={save} />
      <Provider id="github" label="GitHub" st={{ ...st.providers.github, callback_base: st.callback_base }} onSave={save} />
      {err && <p className="lp-error">{err}</p>}
    </>
  );
}
