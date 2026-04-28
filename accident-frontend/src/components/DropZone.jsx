import { useRef, useCallback } from "react";
import "../styles/home.css";

export default function DropZone({ onFile, disabled }) {
  const inputRef  = useRef(null);
  const isDragging = useRef(false);

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      onFile(null, "Please upload a video file.");
      return;
    }
    onFile(f, null);
  }, [onFile]);

  const onDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("drag-over");
    handleFile(e.dataTransfer.files?.[0]);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    e.currentTarget.classList.add("drag-over");
  };

  const onDragLeave = (e) => {
    e.currentTarget.classList.remove("drag-over");
  };

  return (
    <div
      className="dropzone"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        disabled={disabled}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <span className="dz-icon">🎬</span>
      <p className="dz-primary">Drop your dashcam footage here</p>
      <p className="dz-secondary">
        or click to browse &nbsp;·&nbsp; MP4, AVI, MOV supported
      </p>
    </div>
  );
}