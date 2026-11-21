// Credits: the balance, what a Run costs, the packages to buy and recent changes.
// Opens from the account menu, or by itself when a Run needs more credits (402).
import React, { useEffect, useState } from "react";
import { api } from "../../api/client.js";
import { IconClose } from "../ui/Icons.jsx";

const n = (v) => Number(v || 0).toLocaleString("en-US");
const when = (t) => new Date(t * 1000).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

function tierRows(me) {
  const rows = [];
  let from = 0;
  for (const t of me.run_tiers) {
    rows.push([`${n(from)} – ${n(t.up_to_km2)} km²`, t.credits]);
    from = t.up_to_km2;
  }
  const o = me.over_last;
  rows.push([`over ${n(from)} km²`, `+${n(o.credits)} per ${n(o.per_km2)} km²`]);
  return rows;
}

export default function CreditsDialog({ open, onClose, need = null }) {
  const [me, setMe] = useState(null);
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { if (open) { setErr(null); api.billingMe().then(setMe).catch((e) => setErr(e.message)); } }, [open]);
  useEffect(() => {
    if (!open) return undefined;
    const k = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [open, onClose]);
  if (!open) return null;

  async function buy(id) {
    setBusy(id); setErr(null);
    try { window.location.assign((await api.billingCheckout(id)).url); }
    catch (e) { setErr(e.message); setBusy(null); }
  }

  return (
    <div className="set-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="cr-window" role="dialog" aria-modal="true" aria-label="Credits">
        <header className="set-head cr-head">
          <h2>Credits</h2>
          <button type="button" className="panel-close-btn" onClick={onClose} aria-label="Close"><IconClose /></button>
        </header>
        <div className="cr-body">
          {!me ? <p className="rs-dim">{err || "…"}</p> : me.unlimited ? (
            <div className="cr-balance"><span className="cr-figure mono">Unlimited</span><span className="rs-dim">The owner's runs are free</span></div>
          ) : (<>
            <div className="cr-balance">
              <span className="cr-figure mono">{n(me.credits)}</span><span className="rs-dim">credits</span>
            </div>
            {need?.needed != null && (
              <p className="cr-need">This Run needs <b className="mono">{n(need.needed)}</b> credits. Add credits to continue</p>
            )}
            <section className="cr-sec">
              <h3>Add credits</h3>
              <div className="cr-packs">
                {me.packages.map((p, i) => (
                  <button key={p.id} type="button" className={`cr-pack ${i === me.packages.length - 1 ? "best" : ""}`}
                          onClick={() => buy(p.id)} disabled={!me.payments || !!busy}>
                    <span className="cr-pack-credits mono">{n(p.credits)}</span>
                    <span className="rs-dim">credits</span>
                    <span className="cr-pack-price mono">${n(p.usd)}</span>
                    {busy === p.id && <span className="rs-dim">Opening…</span>}
                  </button>
                ))}
              </div>
              {!me.payments && <p className="rs-dim">Card payments are not set up yet. Ask the owner to add credits</p>}
            </section>
          </>)}
          {me && (
            <section className="cr-sec">
              <h3>A Run costs</h3>
              <table className="cr-table">
                <tbody>{tierRows(me).map(([a, c]) => <tr key={a}><td>{a}</td><td className="mono">{typeof c === "number" ? n(c) : c}</td></tr>)}</tbody>
              </table>
              <p className="rs-dim">Charged when a Run starts, given back if it fails</p>
            </section>
          )}
          {me && !me.unlimited && me.history.length > 0 && (
            <section className="cr-sec">
              <h3>Recent</h3>
              <table className="cr-table">
                <tbody>{me.history.slice(0, 8).map((h) => (
                  <tr key={h.id}><td className="mono cr-date">{when(h.created_at)}</td><td>{h.reason}</td>
                    <td className={`mono ${h.delta > 0 ? "cr-plus" : ""}`}>{h.delta > 0 ? "+" : ""}{n(h.delta)}</td></tr>
                ))}</tbody>
              </table>
            </section>
          )}
          {err && me && <p className="lp-error">{err}</p>}
        </div>
      </div>
    </div>
  );
}
