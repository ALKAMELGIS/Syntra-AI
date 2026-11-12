import React, { useState } from "react";
import ModelsDashboard from "./ModelsDashboard.jsx";
import {
  IconGear,
  IconPencil,
  IconDatabase,
  IconPlay,
  IconEye,
} from "./Icons.jsx";

// Mirrors the reference workflow: Setup -> Annotate -> Dataset -> Train -> Detect.
// Each step is a real, honestly-labeled stage of the pipeline — a step that is
// "not built yet" says so plainly instead of faking a working control.
const STEPS = [
  { key: "setup", title: "Setup", icon: IconGear },
  { key: "annotate", title: "Annotate", icon: IconPencil },
  { key: "dataset", title: "Dataset", icon: IconDatabase },
  { key: "train", title: "Train", icon: IconPlay },
  { key: "detect", title: "Detect", icon: IconEye },
];

function StepBadge({ index, state }) {
  // state: "current" | "done" | "pending"
  return (
    <span className={`wizard-step-badge wizard-step-badge-${state}`} aria-hidden="true">
      {state === "done" ? "✓" : index + 1}
    </span>
  );
}

function WizardStepBar({ current, reached, onGo }) {
  return (
    <div className="wizard-steps" role="tablist">
      {STEPS.map((s, i) => {
        const state = i === current ? "current" : i <= reached ? "done" : "pending";
        const Icon = s.icon;
        const clickable = i <= reached;
        return (
          <button
            key={s.key}
            role="tab"
            aria-selected={i === current}
            className={`wizard-step wizard-step-${state}`}
            disabled={!clickable}
            onClick={() => clickable && onGo(i)}
          >
            <StepBadge index={i} state={state} />
            {Icon && <Icon />}
            <span className="wizard-step-title">{s.title}</span>
          </button>
        );
      })}
    </div>
  );
}

function ComingSoon({ title, description, bullets }) {
  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">{title}</p>
      <p className="note" style={{ maxWidth: 640 }}>{description}</p>
      {bullets?.length > 0 && (
        <ul className="wizard-coming-soon-list">
          {bullets.map((b, i) => <li key={i}>{b}</li>)}
        </ul>
      )}
      <p className="note" style={{ marginTop: 14, color: "var(--warn)" }}>
        ⚠ Not built yet — this step is a planned part of the workflow, not a working control.
      </p>
    </div>
  );
}

function AnnotateStep({ onViewChange }) {
  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Annotate</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Object annotation happens directly on the map: use the drawing tools (box, circle, polygon)
        on an uploaded image to mark the objects you want the model to learn.
      </p>
      <button className="run-button" style={{ width: "auto", marginTop: 12, padding: "8px 18px" }}
        onClick={() => onViewChange?.("detect")}>
        Open Full Map to annotate
      </button>
    </div>
  );
}

function DetectStep({ onViewChange }) {
  return (
    <div className="panel wizard-coming-soon">
      <p className="section-title">Detect</p>
      <p className="note" style={{ maxWidth: 640 }}>
        Detection runs on the Full Map: pick an uploaded image and a ready model, then Run Prediction
        from the sidebar. This step will gain a direct link to a model you just trained once the
        Train step below is wired up.
      </p>
      <button className="run-button" style={{ width: "auto", marginTop: 12, padding: "8px 18px" }}
        onClick={() => onViewChange?.("detect")}>
        Open Full Map to detect
      </button>
    </div>
  );
}

export default function ModelSetupWizard({ onViewChange }) {
  const [current, setCurrent] = useState(0);
  const [reached, setReached] = useState(0);

  function go(i) {
    setCurrent(i);
    setReached((r) => Math.max(r, i));
  }

  return (
    <div className="wizard-shell">
      <WizardStepBar current={current} reached={reached} onGo={go} />
      <div className="wizard-body">
        {current === 0 && <ModelsDashboard />}
        {current === 1 && <AnnotateStep onViewChange={onViewChange} />}
        {current === 2 && (
          <ComingSoon
            title="Dataset"
            description="Slices a large orthomosaic into tiles, converts the annotations you drew into YOLO labels, filters out empty/black tiles, samples background tiles, and splits train/val automatically."
            bullets={[
              "Tile size and overlap suggested automatically from your objects' pixel size",
              "Rotation / flip augmentation on the training tiles",
              "Writes data.yaml + dataset_info.json ready for the Train step",
            ]}
          />
        )}
        {current === 3 && (
          <ComingSoon
            title="Train"
            description="Trains a YOLO model on the dataset from the step above, in a separate process so it never freezes the app. Works on CPU, faster on GPU when available."
            bullets={[
              "Epochs, image size, batch size, optimizer, learning rate",
              "Live per-epoch progress: loss, precision, recall, mAP50",
              "Automatic batch-size and worker tuning for your machine's RAM/VRAM",
            ]}
          />
        )}
        {current === 4 && <DetectStep onViewChange={onViewChange} />}
      </div>
      <div className="wizard-nav">
        <button className="chip-button" disabled={current === 0} onClick={() => go(current - 1)}>‹ Back</button>
        <span className="note">Step {current + 1} of {STEPS.length} · {STEPS[current].title}</span>
        <button className="chip-button" disabled={current === STEPS.length - 1} onClick={() => go(current + 1)}>Next ›</button>
      </div>
    </div>
  );
}
