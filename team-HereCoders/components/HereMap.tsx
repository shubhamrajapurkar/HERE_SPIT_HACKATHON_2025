import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Location from 'expo-location';
import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  FlatList,
  Modal,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { WebView } from 'react-native-webview';

const HERE_API_KEY = 'WZwgbIu9MjlypvZgiMPPLpVX8f6aTnl9gqJEHN1TZaU';
const MIN_DISTANCE_THRESHOLD = 5; // meters
const MIN_TIME_INTERVAL = 1000; // milliseconds

function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (val: number) => (val * Math.PI) / 180;
  const R = 6371e3; // earth radius in meters
  const φ1 = toRad(lat1);
  const φ2 = toRad(lat2);
  const Δφ = toRad(lat2 - lat1);
  const Δλ = toRad(lon2 - lon1);

  const a =
    Math.sin(Δφ / 2) ** 2 +
    Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

const activities = [
  { key: 'run', label: 'Run' },
  { key: 'trail_run', label: 'Trail Run' },
  { key: 'walk', label: 'Walk' },
  { key: 'cycle', label: 'Cycle' },
  { key: 'swimming', label: 'Swimming' },
];

export default function HereMap() {
  const webRef = useRef<WebView>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [tracking, setTracking] = useState(false);
  const pathRef = useRef<{ lat: number; lng: number; alt: number }[]>([]);
  const [distance, setDistance] = useState(0);
  const [elevationGain, setElevationGain] = useState(0);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const watcher = useRef<Location.LocationSubscription | null>(null);
  const lastUpdateTime = useRef<number>(0);

  const [selectedActivity, setSelectedActivity] = useState(activities[0]);
  const [modalVisible, setModalVisible] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location permission is required.');
        return;
      }

      watcher.current = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.BestForNavigation,
          timeInterval: 1000,
          distanceInterval: 1,
        },
        (loc) => {
          const currentTime = Date.now();
          if (currentTime - lastUpdateTime.current > MIN_TIME_INTERVAL) {
            lastUpdateTime.current = currentTime;

            setLocation(loc);
            const coords = {
              lat: loc.coords.latitude,
              lng: loc.coords.longitude,
              alt: loc.coords.altitude || 0,
            };

            const jsCode = `window.updateMapLocation(${coords.lat}, ${coords.lng}); true;`;
            webRef.current?.injectJavaScript(jsCode);

            if (tracking) {
              const path = pathRef.current;
              if (path.length === 0) {
                pathRef.current = [coords];
              } else {
                const last = path[path.length - 1];
                const dist = haversineDistance(last.lat, last.lng, coords.lat, coords.lng);
                const elevationDiff = Math.max(0, coords.alt - last.alt);
                if (dist > MIN_DISTANCE_THRESHOLD) {
                  pathRef.current = [...path, coords];
                  setDistance((prev) => prev + dist);
                  setElevationGain((prev) => prev + elevationDiff);
                }
              }
            }
          }
        }
      );
    })();

    return () => {
      watcher.current?.remove();
    };
  }, [tracking]);

  useEffect(() => {
    let interval: number | null = null;
    if (tracking && startTime) {
      interval = setInterval(() => {
        setElapsed((new Date().getTime() - startTime.getTime()) / 1000);
      }, 1000) as unknown as number;
    } else {
      setElapsed(0);
    }
    return () => {
      if (interval !== null) clearInterval(interval);
    };
  }, [tracking, startTime]);

  const saveActivity = async () => {
    const activity = {
      id: Date.now(),
      activityType: selectedActivity.key,
      distance,
      elevationGain,
      elapsed,
      pace: distance > 0 ? (elapsed / 60) / (distance / 1000) : 0,
      path: pathRef.current,
      timestamp: new Date().toISOString(),
    };

    try {
      const storedActivities = await AsyncStorage.getItem('activities');
      const activities = storedActivities ? JSON.parse(storedActivities) : [];
      activities.push(activity);
      await AsyncStorage.setItem('activities', JSON.stringify(activities));
      Alert.alert('Activity saved', `Saved your ${selectedActivity.label} activity.`);
    } catch (e) {
      Alert.alert('Error saving activity', String(e));
    }
  };

  const onStart = () => {
    setTracking(true);
    setDistance(0);
    setElevationGain(0);
    setElapsed(0);
    setStartTime(new Date());
    pathRef.current = [];
  };

  const onStop = () => {
    setTracking(false);
    saveActivity();
  };

  const mapHtml = `<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="initial-scale=1.0, width=device-width" />
  <style>
    html, body, #mapContainer {
      margin: 0; padding: 0; height: 100%; width: 100%;
      font-family: Arial, sans-serif;
    }
  </style>
  <script src="https://js.api.here.com/v3/3.1/mapsjs-core.js"></script>
  <script src="https://js.api.here.com/v3/3.1/mapsjs-service.js"></script>
  <script src="https://js.api.here.com/v3/3.1/mapsjs-ui.js"></script>
  <script src="https://js.api.here.com/v3/3.1/mapsjs-mapevents.js"></script>
  <link rel="stylesheet" href="https://js.api.here.com/v3/3.1/mapsjs-ui.css" />
</head>
<body>
  <div id="mapContainer"></div>
  <script>
    var platform = new H.service.Platform({apikey: '${HERE_API_KEY}'});
    var defaultLayers = platform.createDefaultLayers();

    var map = new H.Map(
      document.getElementById('mapContainer'),
      defaultLayers.raster.normal.map,
      { zoom: 15, center: { lat: 19.1197, lng: 72.9041 } }
    );

    var behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));
    var ui = H.ui.UI.createDefault(map, defaultLayers);

    var marker = new H.map.Marker({ lat: 19.1197, lng: 72.9041 });
    map.addObject(marker);

    window.updateMapLocation = function(lat, lng) {
      marker.setGeometry({ lat: lat, lng: lng });
      map.setCenter({ lat: lat, lng: lng });
    };
  </script>
</body>
</html>`;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.topBar}>
        <Text style={styles.title}>Record Activity</Text>
        <TouchableOpacity
          style={styles.activitySelector}
          onPress={() => setModalVisible(true)}
          activeOpacity={0.7}
        >
          <Text style={styles.activitySelectorText}>{selectedActivity.label} ▼</Text>
        </TouchableOpacity>
      </View>

      <WebView
        ref={webRef}
        originWhitelist={['*']}
        source={{ html: mapHtml }}
        style={{ flex: 1 }}
      />

      <View style={styles.controls}>
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Distance</Text>
            <Text style={styles.statValue}>{(distance / 1000).toFixed(2)} km</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Elevation Gain</Text>
            <Text style={styles.statValue}>{elevationGain.toFixed(1)} m</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Time</Text>
            <Text style={styles.statValue}>
              {Math.floor(elapsed / 60)}:{(Math.floor(elapsed % 60)).toString().padStart(2, '0')}
            </Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Pace</Text>
            <Text style={styles.statValue}>
              {distance > 0
                ? (() => {
                    const paceSecondsPerKm = elapsed / (distance / 1000);
                    const minutes = Math.floor(paceSecondsPerKm / 60);
                    const seconds = Math.floor(paceSecondsPerKm % 60);
                    return `${minutes}:${seconds.toString().padStart(2, '0')} min/km`;
                  })()
                : 'N/A'}
            </Text>
          </View>
        </View>

        <View style={styles.buttons}>
          {!tracking ? (
            <TouchableOpacity style={styles.startButton} onPress={onStart}>
              <Text style={styles.buttonText}>Start</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.stopButton} onPress={onStop}>
              <Text style={styles.buttonText}>Stop</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <Modal
        animationType="slide"
        transparent
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <TouchableOpacity
          style={styles.modalOverlay}
          activeOpacity={1}
          onPressOut={() => setModalVisible(false)}
        >
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Activity</Text>
            <FlatList
              data={activities}
              keyExtractor={(item) => item.key}
              renderItem={({ item }) => (
                <TouchableOpacity
                  style={[
                    styles.activityOption,
                    item.key === selectedActivity.key && styles.activityOptionSelected,
                  ]}
                  onPress={() => {
                    setSelectedActivity(item);
                    setModalVisible(false);
                  }}
                >
                  <Text
                    style={[
                      styles.activityOptionText,
                      item.key === selectedActivity.key && styles.activityOptionTextSelected,
                    ]}
                  >
                    {item.label}
                  </Text>
                </TouchableOpacity>
              )}
            />
          </View>
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  topBar: {
    height: 88,
    backgroundColor: '#fff',
    borderBottomColor: '#ddd',
    borderBottomWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    justifyContent: 'space-between',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    marginTop: 30,
  },
  activitySelector: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    backgroundColor: '#eee',
    borderRadius: 20,
    marginTop: 30,
  },
  activitySelectorText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
  },
  controls: {
    backgroundColor: '#fff',
    borderTopColor: '#ddd',
    borderTopWidth: 1,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 16,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 14,
    color: '#555',
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttons: {
    flexDirection: 'row',
    justifyContent: 'center',
    padding: 16,
  },
  startButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 20,
  },
  stopButton: {
    backgroundColor: '#F44336',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 20,
  },
  buttonText: {
    fontSize: 16,
    color: '#fff',
    fontWeight: 'bold',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#fff',
    padding: 24,
    borderRadius: 10,
    width: '80%',
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 12,
  },
  activityOption: {
    padding: 10,
    width: '100%',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  activityOptionSelected: {
    backgroundColor: '#eee',
  },
  activityOptionText: {
    fontSize: 16,
    textAlign: 'center',
    color: '#333',
  },
  activityOptionTextSelected: {
    color: '#000',
    fontWeight: 'bold',
  },
});
