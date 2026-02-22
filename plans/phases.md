# Implementation Phases v2

## Completed (v1)

### Phase 0: MCCTP Game State ✅
- Basic game state flowing from Minecraft to Python (16 dims)
- `encode_game_state()` in control_policy.py

### Phase 1: Control Recorder ✅
- `control_recorder.py` — records pose + hands + keyboard/mouse actions
- pynput-based capture (to be replaced in Phase 5)

### Phase 2: Dataset & Model ✅
- `control_dataset.py` — windowed dataset with velocity features
- `control_model.py` — ControlTransformer (551 input, 10 output, ~755K params)
- `bootstrap_controls.py` — converted gesture recordings to control format

### Phase 3: Training Pipeline ✅
- `train_controls.py` — training loop with early stopping
- Per-control threshold optimization on validation set
- Combined score metric (F1 + idle accuracy)

### Phase 4: Integration ✅
- `control_policy.py` — live inference wrapper
- `control_bridge.py` — MCCTP command sender
- Basic forward/sprint/sneak/attack/use_item working from bootstrapped data

---

## New Phases (v2 Expansion)

### Phase 5: Mod Expansion — In-Game Control Capture + Extended Game State

**Goal**: Update MCCTP Fabric mod to broadcast resolved input state and extended game state.

**Mod changes needed**:
1. **Resolved input capture** (replaces pynput):
   - `player.input.movementForward` (float: +forward, -backward)
   - `player.input.movementSideways` (float: +left, -right)
   - `player.input.playerInput.jump()`, `.sneak()`, `.sprint()` (bool)
   - `mc.options.attackKey.isPressed()`, `.useKey.isPressed()` (bool)
   - Yaw/pitch deltas per tick
   - `player.getInventory().selectedSlot` (int 0-8)
   - `mc.options.dropKey.isPressed()`, `.swapHandsKey.isPressed()` (bool)
   - Screen open state + cursor position + mouse buttons when GUI open

2. **Extended game state fields** (from 16 → 46 dims):
   - `player.getAttackCooldownProgress()` — attack cooldown
   - `player.getVelocity().y` — vertical velocity
   - `player.getItemUseTimeLeft()` — item use progress
   - `player.getArmor()` — armor value
   - Offhand item classification (shield/food/totem/empty)
   - `mc.crosshairTarget` — entity/block detection + distance
   - Entity scan: nearest hostile distance, relative yaw, count within 16 blocks
   - `player.getStatusEffects()` — speed, slowness, strength, DOT, fire resist
   - `world.getTimeOfDay()`, `player.horizontalCollision`
   - `mc.currentScreen` — screen type detection
   - `player.isSwimming()`, `player.isFallFlying()`, `player.isClimbing()`, `player.isOnFire()`

3. **Updated Python mcctp package** to parse new fields from mod broadcast

**Deliverables**:
- Updated Fabric mod JAR
- Updated mcctp Python package
- Test script verifying all 46 game state dims + resolved inputs

**Milestone**: Can run test script → see all fields updating in real-time while playing.

---

### Phase 6: Recorder v2 — In-Game Controls + 46-dim Game State

**Goal**: Rewrite `control_recorder.py` to use in-game control capture instead of pynput.

**Changes**:
1. Remove pynput dependency entirely
2. Read controls from MCCTP mod broadcast:
   - Binary actions from resolved input fields
   - Look from yaw/pitch deltas
   - Hotbar from slot change detection
   - Inventory cursor/clicks from mouse state when screen open
3. Expand `encode_game_state()` from 16 → 46 dims
4. Record 28-dim control vector per frame
5. Add mode tracking: gameplay vs screen_open

**Recording format** (.npz per session):
```python
{
    'frames': (N, 260),        # pose + hands features
    'controls': (N, 28),       # 28-dim control vector
    'game_state': (N, 46),     # 46-dim game state
    'timestamps': (N,),        # frame timestamps
    'fps': float,              # recording FPS
}
```

**Deliverables**:
- Updated `control_recorder.py`
- Updated `encode_game_state()` (46 dims)
- Verified recordings with all 28 control dims populated

**Milestone**: Record a 5-minute session → verify all controls appear in data.

---

### Phase 7: Model v2 — 5-Head Transformer

**Goal**: Expand model architecture to 5 output heads, 671-dim input, ~3.5M params.

**Changes**:
1. Update `control_model.py`:
   - Input projection: 671 → 256
   - Transformer: 6 layers, 8 heads, d_ff=512
   - 5 output heads: Action(12), Look(2), Hotbar(9), Cursor(2), InvClick(3)
2. Update `control_dataset.py`:
   - Load 28-dim controls
   - Load 46-dim game state
   - Action history: 28 × 5 = 140 dims
   - Generate masks: gameplay_mask, screen_open_mask, hotbar_change_mask
3. Update config format for new dimensions and all 5 head thresholds

**Deliverables**:
- Updated model definition
- Updated dataset with mask generation
- Config schema for v2

**Milestone**: Model instantiates, forward pass produces correct output shapes.

---

### Phase 8: Training v2 — Mode-Aware Masked Loss

**Goal**: Training pipeline for the 5-head model with masked losses.

**Changes**:
1. Update `train_controls.py`:
   - Mode-aware loss computation (apply masks per head)
   - Per-control pos_weight for action + inv_click heads
   - CrossEntropy for hotbar head
   - Separate metrics per head
   - Combined score: weighted average of all head metrics
2. Threshold optimization for all binary controls
3. Updated training log format
4. Validation: per-head F1, look MSE, cursor MSE, idle accuracy

**Deliverables**:
- Updated training script
- Training log with per-head metrics
- Optimized thresholds saved in config

**Milestone**: Train on live-recorded data → all 5 heads show learning curves.

---

### Phase 9: Inference + Bridge v2 — Full Control Output

**Goal**: Live inference wrapper and MCCTP bridge for all 28 controls.

**Changes**:
1. Update `control_policy.py`:
   - 671-dim input assembly
   - 28-dim output parsing from 5 heads
   - Action history ring buffer (28 × 5)
   - Mode switching: gameplay vs inventory based on screen_open
   - Post-processing: thresholds, hysteresis, EMA smoothing, deadzone
2. Update `control_bridge.py`:
   - Send all 28 controls to MCCTP
   - Hotbar slot commands
   - Drop item, swap offhand, open inventory
   - Cursor movement + inventory clicks when screen open
   - Proper held/pulse edge detection for all controls

**Deliverables**:
- Updated inference wrapper
- Updated MCCTP bridge
- Full control output working in real-time

**Milestone**: Play Minecraft for 2 minutes using only body gestures — move, fight, place blocks, switch hotbar, open inventory.

---

### Phase 10: Polish & Optimization

**Goal**: Refine the system for practical daily use.

**Tasks**:
1. **Data augmentation**: Mirror (left/right swap), temporal jitter, noise injection
2. **Curriculum learning**: Start with movement only → add combat → add inventory
3. **User calibration**: Quick calibration routine at session start
4. **Performance profiling**: Ensure <30ms total pipeline latency
5. **Failure recovery**: Auto-release all actions if tracking lost for >0.5s
6. **Session management**: Resume recordings, merge datasets
7. **Visualization**: Debug overlay showing active controls, model confidence, game state

**Milestone**: Reliable enough for extended play sessions (30+ minutes).

---

## Dependency Graph

```
Phase 5 (Mod) ──→ Phase 6 (Recorder) ──→ Phase 7 (Model) ──→ Phase 8 (Training)
                                                                      │
                                                                      ▼
                                              Phase 9 (Inference) ──→ Phase 10 (Polish)
```

Phase 5 is the critical path — everything else depends on the mod broadcasting
resolved inputs and extended game state.
