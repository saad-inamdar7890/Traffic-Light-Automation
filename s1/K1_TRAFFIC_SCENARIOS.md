# K1 Network - 24-Hour Traffic Flow Scenarios

## 📊 Overview

Successfully created comprehensive 24-hour traffic flow patterns for the K1 network with **realistic time-based scenarios**.

---

## 🗺️ Network Structure

### Traffic Light Junctions (9 total)
- **J0**: Central north junction (4-way, high traffic)
- **J1**: South-central junction
- **J5**: Southwest junction
- **J6**: Northwest junction (4-way arterial)
- **J7**: South junction
- **J10**: Southeast junction
- **J11**: Northeast junction (4-way, east access)
- **J12**: Central hub (4-way, critical bottleneck)
- **J22**: East-central junction (4-way arterial)

### Entry/Exit Points (9 total)
- **North Residential**: J14, J15, J16
- **East Commercial**: J17, J18
- **West Mixed**: J13
- **Southwest Industrial**: J19
- **Southeast Industrial**: J23
- **South Industrial**: J8

---

## 🚗 Vehicle Types (5 types)

| Type | Speed | Accel | Length | Distribution | Color |
|------|-------|-------|--------|--------------|-------|
| **Passenger** | 50 km/h | 2.6 m/s² | 5.0 m | 80% | Yellow |
| **Delivery** | 40 km/h | 2.0 m/s² | 6.5 m | 10% | Blue |
| **Truck** | 35 km/h | 1.5 m/s² | 12.0 m | 5% | Gray |
| **Bus** | 40 km/h | 1.8 m/s² | 12.0 m | 3% | Cyan |
| **Emergency** | 70 km/h | 3.5 m/s² | 5.5 m | 2% | Red |

---

## 📅 Traffic Patterns (6 Time Periods)

### **Period 1: Night (00:00 - 07:00)** 
**Duration:** 7 hours | **Intensity:** Very Low (~150 veh/hr)

**Characteristics:**
- Minimal passenger traffic (background)
- Peak freight delivery (02:00-05:00)
- Occasional emergency vehicles
- Low congestion, free-flowing

**Key Routes:**
- Freight circuits (perimeter and south routes)
- Delivery vans (commercial zones)
- Night shift workers (sporadic)

---

### **Period 2: Morning Rush (07:00 - 09:00)**
**Duration:** 2 hours | **Intensity:** Very High (~900 veh/hr) ⚠️

**Characteristics:**
- **Primary Direction:** North (Residential) → East (Commercial)
- Heavy commuter traffic (70% passenger cars)
- Bus service every 10-15 minutes
- High congestion at J0, J11, J12, J22

**Key Routes:**
1. **J14 → J6 → J0 → J11 → J17** (Main arterial - 200 veh/hr)
2. **J15 → J0 → J11 → J17** (Center route - 180 veh/hr)
3. **J16 → J11 → J22 → J18** (Alternative - 150 veh/hr)
4. **J13 → J6 → J21 → J12 → J22 → J18** (West entry - 80 veh/hr)

**Expected Bottlenecks:**
- ⚠️ J0 (4-way, heavy north-south flow)
- ⚠️ J11 (convergence point)
- ⚠️ J12 (central hub)

---

### **Period 3: Midday (09:00 - 17:00)**
**Duration:** 8 hours | **Intensity:** Moderate (~500 veh/hr)

**Characteristics:**
- **Balanced bidirectional traffic**
- High delivery activity (commercial deliveries)
- Regular bus service (every 15-20 min)
- Mixed vehicle types
- Local circulation patterns

**Traffic Distribution:**
- North ↔ East: 40%
- West ↔ East: 25%
- South routes: 20%
- Local loops: 15%

**Key Activities:**
- Commercial deliveries peak
- Emergency vehicles (random, ~5-10/hr)
- Moderate truck traffic
- Even distribution across all junctions

---

### **Period 4: Evening Rush (17:00 - 19:00)**
**Duration:** 2 hours | **Intensity:** Very High (~950 veh/hr) ⚠️

**Characteristics:**
- **Primary Direction:** East (Commercial) → North (Residential)
- **Reverse commute** (opposite of morning)
- Peak congestion period
- Bus service every 10-15 minutes
- Declining delivery traffic

**Key Routes:**
1. **J17 → J11 → J0 → J6 → J14** (Main return - 220 veh/hr)
2. **J17 → J11 → J0 → J15** (Center return - 190 veh/hr)
3. **J18 → J22 → J12 → J0 → J15** (Alternative - 160 veh/hr)
4. **J18 → J22 → J12 → J21 → J6 → J13** (West exit - 85 veh/hr)

**Expected Bottlenecks:**
- ⚠️ J11 (4-way, heavy outflow)
- ⚠️ J22 (east-west arterial)
- ⚠️ J12 (central convergence)

---

### **Period 5: Early Night (19:00 - 22:00)**
**Duration:** 3 hours | **Intensity:** Low (~300 veh/hr)

**Characteristics:**
- Declining passenger traffic
- Increasing freight activity
- Reduced bus service (every 30 min)
- Evening social/entertainment trips
- Transition to night patterns

**Traffic Mix:**
- Passenger: 60%
- Delivery: 25%
- Truck: 12%
- Bus: 3%

---

### **Period 6: Late Night (22:00 - 00:00)**
**Duration:** 2 hours | **Intensity:** Very Low (~100 veh/hr)

**Characteristics:**
- Minimal passenger traffic
- Peak freight delivery period
- Last bus service (hourly)
- Rare emergency vehicles
- Very low congestion

**Dominant Traffic:**
- Freight trucks (60%)
- Delivery vans (30%)
- Late workers (8%)
- Emergency (2%)

---

## 📈 Expected Traffic Statistics (24-Hour Total)

### Total Vehicle Counts
```
Period 1 (Night):        ~1,050 vehicles
Period 2 (Morning):      ~1,800 vehicles
Period 3 (Midday):       ~4,000 vehicles
Period 4 (Evening):      ~1,900 vehicles
Period 5 (Early Night):  ~900 vehicles
Period 6 (Late Night):   ~200 vehicles
------------------------
TOTAL 24H:              ~9,850 vehicles
```

### Peak Hour Analysis
```
Morning Peak (08:00-09:00):  ~950 vehicles
Evening Peak (17:00-18:00):  ~980 vehicles (highest)
Midday Average:              ~500 vehicles/hour
Night Minimum (03:00-04:00): ~100 vehicles
```

### Junction Load Ranking (Expected)
```
1. J12 (Central Hub):     ~2,500 vehicles/day
2. J0 (North Central):    ~2,200 vehicles/day
3. J11 (East Access):     ~2,000 vehicles/day
4. J22 (East Arterial):   ~1,800 vehicles/day
5. J6 (West Arterial):    ~1,500 vehicles/day
6. J1 (South Central):    ~1,200 vehicles/day
7. J10 (Southeast):       ~900 vehicles/day
8. J5 (Southwest):        ~800 vehicles/day
9. J7 (South):            ~600 vehicles/day
```

---

## 🎯 Key Scenarios for RL Training

### Scenario 1: Morning Congestion Management
- **Time:** 07:00-09:00
- **Challenge:** Handle 900 veh/hr north→east flow
- **RL Goal:** Minimize waiting time at J0, J11, J12
- **Coordination:** J0 ↔ J11 signal synchronization

### Scenario 2: Midday Multi-Directional Balance
- **Time:** 09:00-17:00
- **Challenge:** Balance 4-way traffic at all junctions
- **RL Goal:** Maximize throughput, prevent local congestion
- **Coordination:** Network-wide phase optimization

### Scenario 3: Evening Reverse Rush
- **Time:** 17:00-19:00
- **Challenge:** Handle 950 veh/hr east→north flow
- **RL Goal:** Optimize reverse commute (vs morning)
- **Coordination:** J22 ↔ J11 ↔ J0 signal chain

### Scenario 4: Night Freight Optimization
- **Time:** 22:00-05:00
- **Challenge:** Minimize delays for heavy vehicles
- **RL Goal:** Adaptive green times for low traffic
- **Coordination:** Peripheral route optimization (J5, J7, J21)

### Scenario 5: Emergency Response
- **Time:** Random throughout day
- **Challenge:** Priority for emergency vehicles
- **RL Goal:** Clear paths quickly, resume normal flow
- **Coordination:** Upstream signal preemption

### Scenario 6: Peak-to-Off-Peak Transition
- **Time:** 09:00-10:00, 19:00-20:00
- **Challenge:** Adapt from peak to normal patterns
- **RL Goal:** Smooth transition, no sudden changes
- **Coordination:** Gradual phase duration adjustments

---

## 🚀 Next Steps

### 1. Test the Baseline Simulation
```bash
cd s1
sumo-gui k1.sumocfg
```
- Run for 86,400 seconds (24 hours)
- Observe traffic patterns visually
- Identify bottlenecks

### 2. Measure Baseline Metrics
- Average waiting time per junction
- Queue lengths during peak hours
- Throughput (vehicles/hour)
- Travel time for each route
- Congestion hotspots

### 3. Prepare RL Training
- Define state space (queue, wait, phase, neighbor states)
- Define action space (keep, switch, extend phase)
- Implement reward function
- Set up multi-agent framework

### 4. Start Training
- Train for 49 episodes (24h each)
- Monitor convergence
- Compare vs baseline

---

## 📝 File Structure

```
s1/
├── k1.net.xml                  # Network topology (9 TLS junctions)
├── k1_routes_24h.rou.xml       # Traffic flow scenarios (THIS FILE)
├── k1.sumocfg                  # SUMO configuration
└── K1_TRAFFIC_SCENARIOS.md     # This documentation
```

---

## 🔍 Validation Checklist

Before running full training:

- [ ] Test 1-hour simulation (0-3600s) - verify vehicles spawn
- [ ] Test morning rush (25200-32400s) - check congestion
- [ ] Test evening rush (61200-68400s) - check reverse flow
- [ ] Verify all 30+ routes are functional
- [ ] Check vehicle types are spawning correctly
- [ ] Confirm no deadlocks or eternal queues
- [ ] Measure baseline performance metrics

---

**Status:** ✅ Traffic scenarios ready for simulation!

**Estimated Baseline Run Time:** 
- Real-time: 24 hours
- SUMO simulation: ~15-30 minutes (depending on hardware)
- With RL agent: ~1-2 hours per episode

**Ready for:** Phase 2 - RL Training Setup 🚀
