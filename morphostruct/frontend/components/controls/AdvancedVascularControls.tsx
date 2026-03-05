'use client';

import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Plus, Trash2, Info } from 'lucide-react';
import { ScaffoldType } from '@/lib/types/scaffolds';

interface AdvancedVascularControlsProps {
  scaffoldType: ScaffoldType.SPACE_COLONIZATION | ScaffoldType.BIFURCATING_TREE;
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

export function AdvancedVascularControls({ scaffoldType, params, onChange }: AdvancedVascularControlsProps) {
  const isSpaceColonization = scaffoldType === ScaffoldType.SPACE_COLONIZATION;

  // Space Colonization: Inlet Management
  const inlets = params.inlets || [{
    position: [0, 0, 0.001],
    radius: 0.0002,
    direction: [0, 0, -1]
  }];

  const updateInlet = (index: number, field: string, value: any) => {
    const newInlets = [...inlets];
    newInlets[index] = { ...newInlets[index], [field]: value };
    onChange('inlets', newInlets);
  };

  const addInlet = () => {
    onChange('inlets', [...inlets, {
      position: [0, 0, 0.001],
      radius: 0.0002,
      direction: [0, 0, -1]
    }]);
  };

  const removeInlet = (index: number) => {
    if (inlets.length > 1) {
      onChange('inlets', inlets.filter((_: any, i: number) => i !== index));
    }
  };

  // Bifurcating Tree: Root configuration
  const rootPosition = params.root_position || [0, 0, 0.001];
  const rootDirection = params.root_direction || [0, 0, -1];

  const updateRootPosition = (index: number, value: number) => {
    const newPos = [...rootPosition];
    newPos[index] = value;
    onChange('root_position', newPos);
  };

  const updateRootDirection = (index: number, value: number) => {
    const newDir = [...rootDirection];
    newDir[index] = value;
    onChange('root_direction', newDir);
  };

  return (
    <div className="space-y-4">
      <Tabs defaultValue="basic" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="basic">Basic</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
          <TabsTrigger value="radius">Radius</TabsTrigger>
        </TabsList>

        <TabsContent value="basic" className="space-y-4 mt-4">
          {isSpaceColonization ? (
            <>
              {/* Space Colonization: Inlets */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Inlet Ports</Label>
                  <Button onClick={addInlet} size="sm" variant="outline">
                    <Plus className="h-4 w-4 mr-1" />
                    Add
                  </Button>
                </div>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {inlets.map((inlet: any, index: number) => (
                    <div key={index} className="border rounded p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium">Inlet {index + 1}</span>
                        {inlets.length > 1 && (
                          <Button onClick={() => removeInlet(index)} size="sm" variant="ghost">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
                      <div className="grid grid-cols-3 gap-1">
                        {[0, 1, 2].map((i) => (
                          <Input
                            key={i}
                            type="number"
                            step="0.0001"
                            value={inlet.position[i]}
                            onChange={(e) => {
                              const pos = [...inlet.position];
                              pos[i] = parseFloat(e.target.value) || 0;
                              updateInlet(index, 'position', pos);
                            }}
                            className="h-7 text-xs"
                            placeholder={['X','Y','Z'][i]}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Attraction Points */}
              <div>
                <Label className="text-sm">Attraction Points: {params.num_attractors || 50000}</Label>
                <Slider
                  value={[params.num_attractors || 50000]}
                  onValueChange={([val]) => onChange('num_attractors', val)}
                  min={1000}
                  max={200000}
                  step={1000}
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-1">More points = denser network</p>
              </div>

              {/* Max Iterations */}
              <div>
                <Label className="text-sm">Max Iterations: {params.max_iterations || 300}</Label>
                <Slider
                  value={[params.max_iterations || 300]}
                  onValueChange={([val]) => onChange('max_iterations', val)}
                  min={50}
                  max={1000}
                  step={10}
                  className="mt-2"
                />
              </div>
            </>
          ) : (
            <>
              {/* Bifurcating Tree: Root Position */}
              <div>
                <Label className="text-sm mb-2">Root Position (m)</Label>
                <div className="grid grid-cols-3 gap-2">
                  {['X', 'Y', 'Z'].map((label, i) => (
                    <div key={i}>
                      <Label className="text-xs">{label}</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={rootPosition[i]}
                        onChange={(e) => updateRootPosition(i, parseFloat(e.target.value) || 0)}
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Branching Levels */}
              <div>
                <Label className="text-sm">Branching Levels: {params.branching_levels || 5}</Label>
                <Slider
                  value={[params.branching_levels || 5]}
                  onValueChange={([val]) => onChange('branching_levels', val)}
                  min={1}
                  max={10}
                  step={1}
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Total branches: {Math.pow(params.branches_per_node || 2, params.branching_levels || 5)}
                </p>
              </div>

              {/* Branches Per Node */}
              <div>
                <Label className="text-sm">Branches Per Node: {params.branches_per_node || 2}</Label>
                <Slider
                  value={[params.branches_per_node || 2]}
                  onValueChange={([val]) => onChange('branches_per_node', val)}
                  min={2}
                  max={5}
                  step={1}
                  className="mt-2"
                />
              </div>

              {/* Branching Angle */}
              <div>
                <Label className="text-sm">Branching Angle: {params.branching_angle_deg || 35}°</Label>
                <Slider
                  value={[params.branching_angle_deg || 35]}
                  onValueChange={([val]) => onChange('branching_angle_deg', val)}
                  min={10}
                  max={90}
                  step={5}
                  className="mt-2"
                />
              </div>
            </>
          )}
        </TabsContent>

        <TabsContent value="advanced" className="space-y-4 mt-4">
          {isSpaceColonization ? (
            <>
              {/* Bifurcation Controls */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Enable Bifurcation</Label>
                  <Switch
                    checked={params.enable_bifurcation ?? true}
                    onCheckedChange={(checked) => onChange('enable_bifurcation', checked)}
                  />
                </div>
                {params.enable_bifurcation !== false && (
                  <div className="space-y-3 ml-4">
                    <div>
                      <Label className="text-sm">Probability: {params.bifurcation_probability || 0.35}</Label>
                      <Slider
                        value={[params.bifurcation_probability || 0.35]}
                        onValueChange={([val]) => onChange('bifurcation_probability', val)}
                        min={0}
                        max={1}
                        step={0.05}
                        className="mt-2"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Multi-Inlet Mode */}
              <div>
                <Label className="text-sm mb-2">Multi-Inlet Mode</Label>
                <Select
                  value={params.multi_inlet_mode || 'blended'}
                  onValueChange={(val) => onChange('multi_inlet_mode', val)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="blended">Blended (Recommended)</SelectItem>
                    <SelectItem value="partitioned_xy">Partitioned</SelectItem>
                    <SelectItem value="forest">Forest (Separate Trees)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </>
          ) : (
            <>
              {/* Segment Length */}
              <div>
                <Label className="text-sm">Segment Length: {(params.segment_length || 0.0003) * 1000} mm</Label>
                <Slider
                  value={[(params.segment_length || 0.0003) * 1000]}
                  onValueChange={([val]) => onChange('segment_length', val / 1000)}
                  min={0.1}
                  max={2}
                  step={0.05}
                  className="mt-2"
                />
              </div>

              {/* Length Tapering */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Taper Segment Length</Label>
                  <Switch
                    checked={params.taper_segment_length ?? true}
                    onCheckedChange={(checked) => onChange('taper_segment_length', checked)}
                  />
                </div>
                {params.taper_segment_length !== false && (
                  <div className="ml-4">
                    <Label className="text-sm">Taper Factor: {params.length_taper_factor || 0.85}</Label>
                    <Slider
                      value={[params.length_taper_factor || 0.85]}
                      onValueChange={([val]) => onChange('length_taper_factor', val)}
                      min={0.5}
                      max={1}
                      step={0.05}
                      className="mt-2"
                    />
                  </div>
                )}
              </div>

              {/* Variation */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Label>Add Variation</Label>
                  <Switch
                    checked={params.add_variation || false}
                    onCheckedChange={(checked) => onChange('add_variation', checked)}
                  />
                </div>
                {params.add_variation && (
                  <div className="space-y-3 ml-4">
                    <div>
                      <Label className="text-sm">Angle Variation: ±{params.angle_variation_deg || 10}°</Label>
                      <Slider
                        value={[params.angle_variation_deg || 10]}
                        onValueChange={([val]) => onChange('angle_variation_deg', val)}
                        min={0}
                        max={45}
                        step={5}
                        className="mt-2"
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Random Seed */}
          <div>
            <Label className="text-sm">Random Seed</Label>
            <Input
              type="number"
              value={params.random_seed || 42}
              onChange={(e) => onChange('random_seed', parseInt(e.target.value) || 42)}
              className="mt-2"
            />
          </div>
        </TabsContent>

        <TabsContent value="radius" className="space-y-4 mt-4">
          {isSpaceColonization ? (
            <>
              <div>
                <Label className="text-sm">Min Radius: {(params.min_radius || 0.00003) * 1000000} μm</Label>
                <Slider
                  value={[(params.min_radius || 0.00003) * 1000000]}
                  onValueChange={([val]) => onChange('min_radius', val / 1000000)}
                  min={10}
                  max={200}
                  step={5}
                  className="mt-2"
                />
              </div>

              <div>
                <Label className="text-sm">Max Radius: {(params.max_radius || 0.0002) * 1000000} μm</Label>
                <Slider
                  value={[(params.max_radius || 0.0002) * 1000000]}
                  onValueChange={([val]) => onChange('max_radius', val / 1000000)}
                  min={50}
                  max={500}
                  step={10}
                  className="mt-2"
                />
              </div>

              <div>
                <Label className="text-sm">Taper Factor: {params.taper_factor || 0.95}</Label>
                <Slider
                  value={[params.taper_factor || 0.95]}
                  onValueChange={([val]) => onChange('taper_factor', val)}
                  min={0.5}
                  max={1}
                  step={0.01}
                  className="mt-2"
                />
                <p className="text-xs text-muted-foreground mt-1">Child radius = parent × taper</p>
              </div>
            </>
          ) : (
            <>
              <div>
                <Label className="text-sm mb-2">Radius Mode</Label>
                <Select
                  value={params.radius_mode || 'murray'}
                  onValueChange={(val) => onChange('radius_mode', val)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="murray">Murray's Law</SelectItem>
                    <SelectItem value="linear">Linear Taper</SelectItem>
                    <SelectItem value="fixed">Fixed Radius</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {params.radius_mode === 'murray' && (
                <div>
                  <Label className="text-sm">Murray Exponent: {params.murray_exponent || 3}</Label>
                  <Slider
                    value={[params.murray_exponent || 3]}
                    onValueChange={([val]) => onChange('murray_exponent', val)}
                    min={2}
                    max={4}
                    step={0.1}
                    className="mt-2"
                  />
                  <p className="text-xs text-muted-foreground mt-1">3.0 = optimal for laminar flow</p>
                </div>
              )}

              <div>
                <Label className="text-sm">Root Radius: {(params.root_radius || 0.0002) * 1000} mm</Label>
                <Slider
                  value={[(params.root_radius || 0.0002) * 1000]}
                  onValueChange={([val]) => onChange('root_radius', val / 1000)}
                  min={0.05}
                  max={1}
                  step={0.01}
                  className="mt-2"
                />
              </div>

              <div>
                <Label className="text-sm">Min Terminal Radius: {(params.min_terminal_radius || 0.00003) * 1000000} μm</Label>
                <Slider
                  value={[(params.min_terminal_radius || 0.00003) * 1000000]}
                  onValueChange={([val]) => onChange('min_terminal_radius', val / 1000000)}
                  min={10}
                  max={200}
                  step={5}
                  className="mt-2"
                />
              </div>
            </>
          )}

          {/* Mesh Resolution */}
          <div>
            <Label className="text-sm">Radial Resolution: {params.radial_resolution || 12}</Label>
            <Slider
              value={[params.radial_resolution || 12]}
              onValueChange={([val]) => onChange('radial_resolution', val)}
              min={6}
              max={32}
              step={2}
              className="mt-2"
            />
            <p className="text-xs text-muted-foreground mt-1">Segments around vessel circumference</p>
          </div>
        </TabsContent>
      </Tabs>

      {/* Info Box */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded p-3">
        <div className="flex gap-2">
          <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-blue-900 dark:text-blue-100">
            {isSpaceColonization ? (
              <>
                <strong>Space Colonization:</strong> Organic vascular growth driven by tissue attraction points.
                Mimics natural angiogenesis. Use multiple inlets for complex networks.
              </>
            ) : (
              <>
                <strong>Bifurcating Tree:</strong> Regular geometric tree with configurable branching.
                Murray's law ensures optimal flow distribution. Good for controlled geometries.
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
