'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Center, Environment } from '@react-three/drei';
import { Suspense, useState } from 'react';
import { ScaffoldMesh } from './ScaffoldMesh';
import { ViewControls } from './ViewControls';
import { VascularOverlay } from './VascularOverlay';

interface ViewportProps {
  meshData?: {
    vertices: number[];
    indices: number[];
    normals?: number[];
  };
  isLoading?: boolean;
}

export function Viewport({ meshData, isLoading = false }: ViewportProps) {
  const [viewMode, setViewMode] = useState<'normal' | 'wireframe' | 'xray'>('normal');
  const [showGrid, setShowGrid] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);

  return (
    <div className="relative w-full h-full min-h-[400px] bg-slate-900 rounded-lg overflow-hidden">
      {/* View controls overlay */}
      <ViewControls
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        showGrid={showGrid}
        onShowGridChange={setShowGrid}
        autoRotate={autoRotate}
        onAutoRotateChange={setAutoRotate}
      />

      {/* Vascular network statistics overlay */}
      <VascularOverlay />

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
          <div className="flex flex-col items-center gap-2">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-slate-300">Generating scaffold...</span>
          </div>
        </div>
      )}

      {/* Three.js Canvas */}
      <Canvas
        camera={{ position: [30, 30, 30], fov: 50 }}
        gl={{ antialias: true }}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} />
          <directionalLight position={[-10, -10, -5]} intensity={0.3} />

          {/* Environment for reflections */}
          <Environment preset="studio" />

          {/* Grid */}
          {showGrid && (
            <Grid
              args={[100, 100]}
              cellSize={5}
              cellColor="#334155"
              sectionColor="#475569"
              fadeDistance={100}
              position={[0, -0.01, 0]}
            />
          )}

          {/* Scaffold mesh */}
          <Center>
            <ScaffoldMesh
              meshData={meshData}
              viewMode={viewMode}
            />
          </Center>

          {/* Controls */}
          <OrbitControls
            makeDefault
            autoRotate={autoRotate}
            autoRotateSpeed={2}
            enableDamping
            dampingFactor={0.05}
          />
        </Suspense>
      </Canvas>

      {/* Empty state */}
      {!meshData && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <p className="text-slate-500">
            Generate a scaffold to preview it here
          </p>
        </div>
      )}
    </div>
  );
}
