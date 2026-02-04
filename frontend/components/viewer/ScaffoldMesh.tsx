'use client';

import { useMemo } from 'react';
import * as THREE from 'three';

interface ScaffoldMeshProps {
  meshData?: {
    vertices: number[];
    indices: number[];
    normals?: number[];
  };
  viewMode: 'normal' | 'wireframe' | 'xray';
}

export function ScaffoldMesh({ meshData, viewMode }: ScaffoldMeshProps) {
  const geometry = useMemo(() => {
    if (!meshData || !meshData.vertices.length) return null;

    const geom = new THREE.BufferGeometry();

    // Set vertices
    const positions = new Float32Array(meshData.vertices);
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Set indices if provided
    if (meshData.indices && meshData.indices.length) {
      geom.setIndex(meshData.indices);
    }

    // Calculate normals if not provided
    if (meshData.normals && meshData.normals.length) {
      const normals = new Float32Array(meshData.normals);
      geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    } else {
      geom.computeVertexNormals();
    }

    return geom;
  }, [meshData]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry}>
      {viewMode === 'normal' && (
        <meshStandardMaterial
          color="#3b82f6"
          metalness={0.2}
          roughness={0.6}
          side={THREE.DoubleSide}
        />
      )}
      {viewMode === 'wireframe' && (
        <meshBasicMaterial
          color="#60a5fa"
          wireframe
          transparent
          opacity={0.8}
        />
      )}
      {viewMode === 'xray' && (
        <meshBasicMaterial
          color="#3b82f6"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      )}
    </mesh>
  );
}
