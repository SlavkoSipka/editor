"use client";

import { useCallback, useState } from "react";
import { Upload, Video } from "lucide-react";

type Props = {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
};

export function UploadZone({ onFileSelected, selectedFile }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("video/")) {
        onFileSelected(file);
      }
    },
    [onFileSelected],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={`relative rounded-2xl border-2 border-dashed transition-all p-12 text-center cursor-pointer
        ${dragOver ? "border-indigo-400 bg-indigo-500/5" : "border-neutral-800 hover:border-neutral-700"}
      `}
    >
      <input
        type="file"
        accept="video/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelected(file);
        }}
        className="absolute inset-0 opacity-0 cursor-pointer"
      />

      <div className="flex flex-col items-center gap-3 pointer-events-none">
        {selectedFile ? (
          <>
            <Video className="w-10 h-10 text-indigo-400" />
            <div className="text-base font-medium">{selectedFile.name}</div>
            <div className="text-sm text-neutral-500">
              {(selectedFile.size / 1024 / 1024).toFixed(1)} MB · Click to change
            </div>
          </>
        ) : (
          <>
            <Upload className="w-10 h-10 text-neutral-500" />
            <div className="text-base font-medium">Drop a video here</div>
            <div className="text-sm text-neutral-500">
              MP4, MOV up to 500 MB · 3 minutes max
            </div>
          </>
        )}
      </div>
    </div>
  );
}
