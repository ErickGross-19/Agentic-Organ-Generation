'use client';

import React, { Suspense, useState, useEffect } from 'react';

// Lazy load Three.js components only when mounted (client-side only)
let Canvas: any = null;
let OrbitControls: any = null;
let Grid: any = null;
let Center: any = null;
let Environment: any = null;

interface ViewportProps {
  meshData?: {
    vertices: number[];
    indices: number[];
    normals?: number[];
  };
  isLoading?: boolean;
}

export function ViewportClient({ meshData, isLoading }: ViewportProps) {
  const [showGrid, setShowGrid] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [threejsLoaded, setThreejsLoaded] = useState(false);
  const [ScaffoldMesh, setScaffoldMesh] = useState<any>(null);
  const [ViewControls, setViewControls] = useState<any>(null);
  const [VascularOverlay, setVascularOverlay] = useState<any>(null);

  useEffect(() => {
    // Only load Three.js on the client side
    if (typeof window !== 'undefined') {
      console.log('ViewportClient: Starting to load Three.js modules...');

      Promise.all([
        import('@react-three/fiber').then(mod => {
          console.log('ViewportClient: Loaded @react-three/fiber');
          Canvas = mod.Canvas;
        }).catch(err => {
          console.error('ViewportClient: Failed to load @react-three/fiber:', err);
          throw err;
        }),
        import('@react-three/drei').then(mod => {
          console.log('ViewportClient: Loaded @react-three/drei');
          OrbitControls = mod.OrbitControls;
          Grid = mod.Grid;
          Center = mod.Center;
          Environment = mod.Environment;
        }).catch(err => {
          console.error('ViewportClient: Failed to load @react-three/drei:', err);
          throw err;
        }),
        import('./ScaffoldMesh').then(mod => {
          console.log('ViewportClient: Loaded ScaffoldMesh');
          setScaffoldMesh(() => mod.ScaffoldMesh);
        }).catch(err => {
          console.error('ViewportClient: Failed to load ScaffoldMesh:', err);
          throw err;
        }),
        import('./ViewControls').then(mod => {
          console.log('ViewportClient: Loaded ViewControls');
          setViewControls(() => mod.ViewControls);
        }).catch(err => {
          console.error('ViewportClient: Failed to load ViewControls:', err);
          throw err;
        }),
        import('./VascularOverlay').then(mod => {
          console.log('ViewportClient: Loaded VascularOverlay');
          setVascularOverlay(() => mod.VascularOverlay);
        }).catch(err => {
          console.error('ViewportClient: Failed to load VascularOverlay:', err);
          throw err;
        }),
      ]).then(() => {
        console.log('ViewportClient: All modules loaded successfully!');
        setThreejsLoaded(true);
      }).catch(err => {
        console.error('ViewportClient: Failed to load Three.js modules:', err);
        console.error('ViewportClient: Error details:', err.message, err.stack);
      });
    }
  }, []);

  if (!threejsLoaded || !Canvas) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-900/50 rounded-lg border border-slate-700">
        <div className="text-slate-400">Loading 3D viewport...</div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
      {ViewControls && (
        <ViewControls
          showGrid={showGrid}
          onShowGridChange={setShowGrid}
          wireframe={wireframe}
          onWireframeChange={setWireframe}
          autoRotate={autoRotate}
          onAutoRotateChange={setAutoRotate}
        />
      )}

      {VascularOverlay && <VascularOverlay />}

      {isLoading && (
        <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <div className="text-slate-300 font-medium">Generating scaffold...</div>
          </div>
        </div>
      )}

      {!meshData && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-slate-400">
            <p className="text-lg mb-2">No scaffold generated yet</p>
            <p className="text-sm">Configure parameters and click Generate</p>
          </div>
        </div>
      )}

      <Canvas
        camera={{ position: [15, 15, 15], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <color attach="background" args={['#0f172a']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <directionalLight position={[-10, -10, -5]} intensity={0.5} />

        <Suspense fallback={null}>
          {meshData && ScaffoldMesh && (
            <Center>
              <ScaffoldMesh meshData={meshData} wireframe={wireframe} />
            </Center>
          )}

          {showGrid && Grid && (
            <Grid
              args={[20, 20]}
              cellSize={1}
              cellThickness={0.5}
              cellColor="#334155"
              sectionSize={5}
              sectionThickness={1}
              sectionColor="#475569"
              fadeDistance={30}
              fadeStrength={1}
              infiniteGrid={false}
            />
          )}

          {Environment && <Environment preset="studio" />}
        </Suspense>

        {OrbitControls && (
          <OrbitControls
            makeDefault
            autoRotate={autoRotate}
            autoRotateSpeed={1}
            enableDamping
            dampingFactor={0.05}
          />
        )}
      </Canvas>
    </div>
  );
}
