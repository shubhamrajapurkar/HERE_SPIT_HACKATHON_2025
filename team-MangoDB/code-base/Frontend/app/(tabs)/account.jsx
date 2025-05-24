import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Platform,
  Linking,
  Vibration,
  Image,
  ActivityIndicator,
  TextInput,
  ScrollView,
  Modal,
  Dimensions,
} from "react-native";
import { Accelerometer } from "expo-sensors";
import * as Location from "expo-location";
import axios from "axios";
import { Audio } from "expo-av";
import { MaterialIcons, FontAwesome, Ionicons } from "@expo/vector-icons";
import MapView, { Marker, PROVIDER_GOOGLE, Polyline } from 'react-native-maps';

const { width, height } = Dimensions.get('window');

const App = () => {
  // State variables
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [crashDetected, setCrashDetected] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const [location, setLocation] = useState(null);
  const [serviceRoutes, setServiceRoutes] = useState({});
  const [accelerometerData, setAccelerometerData] = useState({
    x: 0,
    y: 0,
    z: 0,
    force: 0,
  });
  const [message, setMessage] = useState("Monitoring accelerometer data...");
  const [capturedImageUri, setCapturedImageUri] = useState(null);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [audioRecording, setAudioRecording] = useState(null);
  const [emergencyTranscript, setEmergencyTranscript] = useState("");
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [manualTranscriptInput, setManualTranscriptInput] = useState("");
  const [emergencyResponse, setEmergencyResponse] = useState(null);
  const [showResponseModal, setShowResponseModal] = useState(false);
  const [nearbyServices, setNearbyServices] = useState(null);

  // Refs
  const countdownIntervalRef = useRef(null);
  const crashTimeoutRef = useRef(null);
  const accelerometerSubscriptionRef = useRef(null);
  const locationSubscriptionRef = useRef(null);

  // Constants
  const CRASH_THRESHOLD_FORCE = 2.5;
  const CRASH_THRESHOLD_CHANGE = 1.0;
  const ACCELEROMETER_UPDATE_INTERVAL = 200;
  const CRASH_ALERT_DELAY = 10000;

  const forceHistory = useRef([]);
  const HISTORY_SIZE = 5;

  // Process accelerometer data
  const processAccelerometerData = useCallback(
    ({ x, y, z }) => {
      const force = Math.sqrt(x * x + y * y + z * z);
      setAccelerometerData({ x, y, z, force });

      forceHistory.current.push(force);
      if (forceHistory.current.length > HISTORY_SIZE) {
        forceHistory.current.shift();
      }

      if (!crashDetected) {
        if (forceHistory.current.length === HISTORY_SIZE) {
          const averagePreviousForce =
            forceHistory.current
              .slice(0, HISTORY_SIZE - 1)
              .reduce((sum, val) => sum + val, 0) /
            (HISTORY_SIZE - 1);
          const currentForce = forceHistory.current[HISTORY_SIZE - 1];

          if (
            currentForce > CRASH_THRESHOLD_FORCE &&
            Math.abs(currentForce - averagePreviousForce) >
            CRASH_THRESHOLD_CHANGE
          ) {
            handleCrash();
          }
        }
      }
    },
    [crashDetected]
  );

  // Setup monitoring
  useEffect(() => {
    const setupMonitoring = async () => {
      let { status: accelStatus } =
        await Accelerometer.requestPermissionsAsync();
      if (accelStatus !== "granted") {
        Alert.alert(
          "Permission Denied",
          "Accelerometer permission is required for crash detection."
        );
        setIsMonitoring(false);
        setMessage("Accelerometer permission denied. Monitoring stopped.");
        return;
      }

      let { status: locationStatus } =
        await Location.requestForegroundPermissionsAsync();
      if (locationStatus !== "granted") {
        Alert.alert(
          "Permission Denied",
          "Location permission is required to provide location in case of emergency."
        );
      } else {
        locationSubscriptionRef.current = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.High,
            timeInterval: 5000,
            distanceInterval: 10,
          },
          (newLocation) => {
            setLocation(newLocation.coords);
          }
        );
        const initialLocation = await Location.getCurrentPositionAsync({});
        setLocation(initialLocation.coords);
      }

      Accelerometer.setUpdateInterval(ACCELEROMETER_UPDATE_INTERVAL);
      accelerometerSubscriptionRef.current = Accelerometer.addListener(
        processAccelerometerData
      );

      setMessage("Monitoring accelerometer and location data...");
    };

    const stopMonitoring = () => {
      if (accelerometerSubscriptionRef.current) {
        accelerometerSubscriptionRef.current.remove();
        accelerometerSubscriptionRef.current = null;
      }
      if (locationSubscriptionRef.current) {
        locationSubscriptionRef.current.remove();
        locationSubscriptionRef.current = null;
      }
      setMessage("Monitoring stopped.");
    };

    if (isMonitoring) {
      setupMonitoring();
    } else {
      stopMonitoring();
    }

    return () => stopMonitoring();
  }, [isMonitoring, processAccelerometerData]);

  // Simulate taking a picture
  const simulateTakePicture = useCallback(() => {
    const timestamp = new Date().getTime();
    setCapturedImageUri(
      `https://placehold.co/200x150/FF0000/FFFFFF?text=Crash+Pic+${timestamp}`
    );
  }, []);

  // Audio recording functions
  const startEmergencyAudioRecording = async () => {
    try {
      if (audioRecording) {
        await audioRecording.stopAndUnloadAsync();
        setAudioRecording(null);
      }

      const { status: audioStatus } = await Audio.requestPermissionsAsync();
      if (audioStatus !== 'granted') {
        Alert.alert("Permission Denied", "Audio recording permission is required for voice messages.");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording: newRecording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setAudioRecording(newRecording);
      setIsRecordingAudio(true);
      setMessage("Please speak about the emergency...");
      Vibration.vibrate(200);
    } catch (err) {
      console.error("Failed to start emergency audio recording", err);
      Alert.alert(
        "Recording Error",
        "Could not start audio recording. " + err.message
      );
      sendEmergencyAlert(
        location,
        "Audio recording failed. No transcript available."
      );
      setMessage("Audio recording failed, sending alert without transcript.");
    }
  };
  const fetchRoute = async (startLat, startLng, endLat, endLng, serviceType) => {
    try {
      // HERE Routing API v8
      const apiKey = process.env.EXPO_PUBLIC_HERE_API_KEY;
      if (!apiKey) {
        console.error("HERE API key not set in .env");
        return;
      }
      const url = `https://router.hereapi.com/v8/routes?transportMode=car&origin=${startLat},${startLng}&destination=${endLat},${endLng}&return=polyline&apikey=${apiKey}`;

      const response = await axios.get(url);

      console.log(`Route data for ${serviceType}:`, response.data);

      if (
        response.data &&
        response.data.routes &&
        response.data.routes[0] &&
        response.data.routes[0].sections &&
        response.data.routes[0].sections[0] &&
        response.data.routes[0].sections[0].polyline
      ) {
        setServiceRoutes(prev => ({
          ...prev,
          [serviceType]: response.data.routes[0].sections[0].polyline
        }));
      }
    } catch (error) {
      console.error(`Error fetching route for ${serviceType}:`, error);
    }
  };
  const stopEmergencyAudioRecording = async () => {
    try {
      setIsRecordingAudio(false);
      setIsTranscribing(true);
      setMessage("Stopping recording and transcribing...");
      Vibration.vibrate(300);

      await audioRecording.stopAndUnloadAsync();
      const uri = audioRecording.getURI();
      setAudioRecording(null);

      await transcribeEmergencyAudio(uri);
    } catch (err) {
      console.error("Failed to stop emergency audio recording", err);
      Alert.alert(
        "Recording Error",
        "Could not stop audio recording. " + err.message
      );
      setIsTranscribing(false);
      sendEmergencyAlert(
        location,
        "Audio recording stop failed. No transcript available."
      );
      setMessage(
        "Audio recording stop failed, sending alert without transcript."
      );
    }
  };

  // Transcribe audio
  const transcribeEmergencyAudio = async (audioUri) => {
    try {
      setIsTranscribing(true);
      setMessage("Transcribing emergency details...");

      const formData = new FormData();
      const fileUri =
        Platform.OS === "android" ? audioUri : audioUri.replace("file://", "");

      formData.append("file", {
        uri: fileUri,
        type: "audio/m4a",
        name: "emergency_recording.m4a",
      });

      const backendUrl = process.env.EXPO_PUBLIC_BACKEND_IP;
      if (!backendUrl) {
        throw new Error("EXPO_PUBLIC_BACKEND_IP is not set in your .env file.");
      }
      const transcriptionUrl = `${backendUrl}/transcribe`;

      const response = await fetch(transcriptionUrl, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Transcription server response error:", errorText);
        throw new Error(`Transcription server error: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      if (!result.text) {
        throw new Error("Invalid transcription response format: Missing text.");
      }

      setEmergencyTranscript(result.text);
      setIsTranscribing(false);
      setMessage("Transcription complete. Sending alert...");
      sendEmergencyAlert(location, result.text);
    } catch (error) {
      console.error("Transcription error:", error);
      setIsTranscribing(false);
      Alert.alert(
        "Transcription Failed",
        "Could not transcribe audio. " + error.message
      );
      sendEmergencyAlert(
        location,
        "No detailed transcript available due to transcription error."
      );
      setMessage("Transcription failed, sending alert with default message.");
    }
  };

  // Handle manual send
  const handleManualSend = () => {
    if (manualTranscriptInput.trim()) {
      sendEmergencyAlert(location, manualTranscriptInput.trim());
      setManualTranscriptInput("");
    } else {
      Alert.alert("No message", "Please type an emergency message or record audio.");
    }
  };

  // Handle crash
  const handleCrash = useCallback(async () => {
    if (crashDetected) return;

    setCrashDetected(true);
    Vibration.vibrate(500);
    setMessage("Crash Detected! Alerting emergency services...");
    setCountdown(10);
    simulateTakePicture();
    setEmergencyTranscript("");
    setManualTranscriptInput("");

    countdownIntervalRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(countdownIntervalRef.current);
          if (!isRecordingAudio && !isTranscribing) {
            startEmergencyAudioRecording();
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    crashTimeoutRef.current = setTimeout(() => {
      if (crashDetected && !isRecordingAudio && !isTranscribing) {
        clearInterval(countdownIntervalRef.current);
        startEmergencyAudioRecording();
      }
    }, CRASH_ALERT_DELAY);
  }, [crashDetected, location, simulateTakePicture, isRecordingAudio, isTranscribing]);

  // Send emergency alert
  const sendEmergencyAlert = (coords, transcriptToSend) => {
    const backendUrl = process.env.EXPO_PUBLIC_BACKEND_IP;
    if (!backendUrl) {
      Alert.alert("Configuration Error", "EXPO_PUBLIC_BACKEND_IP is not set in your .env file.");
      resetDetection();
      return;
    }
    const processtranscriptionUrl = `${backendUrl}/process_transcript`;

    axios
      .post(processtranscriptionUrl, {
        transcript: transcriptToSend,
        lat: coords ? coords.latitude : 0,
        lng: coords ? coords.longitude : 0,
      })
      .then((res) => {
        const data = res.data;
        console.log('Emergency Response Data:', res.data);
        setEmergencyResponse(data);
        setNearbyServices(data.closest_nearby_services);
        
        // Fetch routes for each service
        Object.entries(data.closest_nearby_services).forEach(([type, service]) => {
          if (service && service.latitude && service.longitude) {
            fetchRoute(
              coords.latitude,
              coords.longitude,
              service.latitude,
              service.longitude,
              type
            );
          }
        });
        
        setShowResponseModal(true);
        
        setMessage(`🚨 Emergency Alert Dispatched!`);
        setCrashDetected(false);
        clearInterval(countdownIntervalRef.current);
        clearTimeout(crashTimeoutRef.current);
        forceHistory.current = [];
        setIsMonitoring(true);
        setCapturedImageUri(null);
        setEmergencyTranscript("");
        setManualTranscriptInput("");
      })
      .catch((error) => {
        console.error("Error making API request:", error);
        Alert.alert(
          "Error",
          "Failed to send emergency alert. Please try again."
        );
        resetDetection();
      });
  };

  // Cancel alert
  const cancelAlert = () => {
    clearInterval(countdownIntervalRef.current);
    clearTimeout(crashTimeoutRef.current);
    if (isRecordingAudio && audioRecording) {
      audioRecording.stopAndUnloadAsync().catch((e) =>
        console.error("Error stopping audio on cancel:", e)
      );
    }
    setIsRecordingAudio(false);
    setIsTranscribing(false);
    setMessage("Emergency alert cancelled.");
    resetDetection();
  };
// Add this function to decode the polyline
const decodePolyline = (str) => {
  var index = 0,
      lat = 0,
      lng = 0,
      coordinates = [],
      shift = 0,
      result = 0,
      byte = null,
      latitude_change,
      longitude_change;

  // Coordinates have variable length when encoded, so just keep
  // track of whether we've hit the end of the string. In each
  // loop iteration, a single coordinate is decoded.
  while (index < str.length) {
    // Reset shift, result, and byte
    byte = null;
    shift = 0;
    result = 0;

    do {
      byte = str.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);

    latitude_change = ((result & 1) ? ~(result >> 1) : (result >> 1));

    shift = result = 0;

    do {
      byte = str.charCodeAt(index++) - 63;
      result |= (byte & 0x1f) << shift;
      shift += 5;
    } while (byte >= 0x20);

    longitude_change = ((result & 1) ? ~(result >> 1) : (result >> 1));

    lat += latitude_change;
    lng += longitude_change;

    coordinates.push([lat / 1E5, lng / 1E5]);
  }

  return coordinates;
};
  // Reset detection
  const resetDetection = () => {
    setCrashDetected(false);
    setCountdown(10);
    forceHistory.current = [];
    setIsMonitoring(true);
    setMessage("Monitoring accelerometer data...");
    setCapturedImageUri(null);
    setEmergencyTranscript("");
    setManualTranscriptInput("");
  };

  // Cleanup
  useEffect(() => {
    return () => {
      clearInterval(countdownIntervalRef.current);
      clearTimeout(crashTimeoutRef.current);
      if (accelerometerSubscriptionRef.current) {
        accelerometerSubscriptionRef.current.remove();
      }
      if (locationSubscriptionRef.current) {
        locationSubscriptionRef.current.remove();
      }
      if (audioRecording?.getStatusAsync) {
        audioRecording
          .getStatusAsync()
          .then((status) => {
            if (!status.isLoaded) return;
            return audioRecording.stopAndUnloadAsync();
          })
          .catch((e) => {
            console.log("Audio cleanup:", e.message);
          });
      }
    };
  }, [audioRecording]);

  // Render service marker
  const renderServiceMarker = (service, type) => {
    if (!service || !service.latitude || !service.longitude) return null;
    
    let icon, color;
    switch (type) {
      case 'police':
        icon = 'shield';
        color = '#2E86C1';
        break;
      case 'hospital':
        icon = 'hospital';
        color = '#E74C3C';
        break;
      case 'firebrigade':
        icon = 'fire';
        color = '#E67E22';
        break;
      default:
        icon = 'map-marker';
        color = '#3498DB';
    }

    return (
      <Marker
        key={`${type}-${service.latitude}-${service.longitude}`}
        coordinate={{ 
          latitude: service.latitude, 
          longitude: service.longitude 
        }}
        title={service.name || type}
        description={`${service.distance_meters.toFixed(0)}m away`}
      >
        <View style={[styles.markerContainer, { backgroundColor: color }]}>
          <FontAwesome name={icon} size={24} color="white" />
        </View>
      </Marker>
    );
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollViewContent}>
        <View style={styles.card}>
          <View style={styles.header}>
            <Ionicons name="alert-circle" size={32} color="#E74C3C" />
            <Text style={styles.title}>Crash Detection System</Text>
          </View>

          {/* Status Bar */}
          <View style={styles.statusBar}>
            <View style={[styles.statusIndicator, { backgroundColor: isMonitoring ? '#2ECC71' : '#E74C3C' }]}>
              <Text style={styles.statusText}>{isMonitoring ? 'ACTIVE' : 'INACTIVE'}</Text>
            </View>
            <Text style={styles.forceText}>Force: {accelerometerData.force.toFixed(2)} G</Text>
          </View>

          {/* Input Section */}
          <View style={styles.inputSection}>
            <Text style={styles.sectionTitle}>Emergency Message</Text>
            <View style={styles.textInputContainer}>
              <TextInput
                style={styles.textInput}
                placeholder="Type emergency message..."
                placeholderTextColor="#888"
                value={manualTranscriptInput}
                onChangeText={setManualTranscriptInput}
                editable={!isRecordingAudio && !isTranscribing && !crashDetected}
                multiline
              />
              <TouchableOpacity
                style={[styles.voiceButton, isRecordingAudio && styles.recordingButton]}
                onPress={isRecordingAudio ? stopEmergencyAudioRecording : startEmergencyAudioRecording}
                disabled={isTranscribing || crashDetected}
              >
                <MaterialIcons 
                  name={isRecordingAudio ? "stop" : "mic"} 
                  size={24} 
                  color="#FFF" 
                />
              </TouchableOpacity>
            </View>
            <TouchableOpacity
              onPress={handleManualSend}
              style={styles.sendButton}
              disabled={isRecordingAudio || isTranscribing || !manualTranscriptInput.trim() || crashDetected}
            >
              <Text style={styles.buttonText}>Send Emergency Alert</Text>
            </TouchableOpacity>
          </View>

          {/* Message Display */}
          <View style={styles.messageContainer}>
            <Text style={styles.messageText}>{message}</Text>
            {emergencyTranscript && !isTranscribing && (
              <View style={styles.transcriptBox}>
                <Text style={styles.transcriptText}>{emergencyTranscript}</Text>
              </View>
            )}
          </View>

          {/* Crash Alert UI */}
          {crashDetected && (
            <View style={styles.alertBox}>
              {isRecordingAudio ? (
                <View style={styles.recordingContainer}>
                  <View style={styles.recordingIndicator}>
                    <ActivityIndicator size="large" color="#E74C3C" />
                    <MaterialIcons name="mic" size={32} color="#E74C3C" />
                  </View>
                  <Text style={styles.recordingText}>
                    Recording emergency message... ({countdown}s)
                  </Text>
                  <TouchableOpacity
                    onPress={stopEmergencyAudioRecording}
                    style={styles.stopRecordingButton}
                  >
                    <Text style={styles.buttonText}>Stop & Send</Text>
                  </TouchableOpacity>
                </View>
              ) : isTranscribing ? (
                <View style={styles.recordingContainer}>
                  <ActivityIndicator size="large" color="#E74C3C" />
                  <Text style={styles.recordingText}>Processing your message...</Text>
                </View>
              ) : (
                <>
                  <Text style={styles.alertTitle}>
                    Emergency Alert in {countdown} seconds...
                  </Text>
                  {location && (
                    <Text style={styles.alertLocation}>
                      <Ionicons name="location" size={16} color="#E74C3C" />{' '}
                      {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
                    </Text>
                  )}
                </>
              )}

              {capturedImageUri && (
                <View style={styles.imagePreviewContainer}>
                  <Image
                    source={{ uri: capturedImageUri }}
                    style={styles.capturedImage}
                  />
                </View>
              )}
              
              <TouchableOpacity 
                onPress={cancelAlert} 
                style={styles.cancelButton}
              >
                <Text style={styles.buttonText}>Cancel Alert</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Location Section */}
          {location && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Current Location</Text>
              <MapView
                style={styles.map}
                provider={PROVIDER_GOOGLE}
                initialRegion={{
                  latitude: location.latitude,
                  longitude: location.longitude,
                  latitudeDelta: 0.005,
                  longitudeDelta: 0.005,
                }}
                scrollEnabled={true}
                zoomEnabled={true}
              >
                <Marker
                  coordinate={{ latitude: location.latitude, longitude: location.longitude }}
                  title="Your Location"
                  description="Potential emergency site"
                >
                  <View style={[styles.markerContainer, { backgroundColor: '#E74C3C' }]}>
                    <Ionicons name="alert-circle" size={20} color="white" />
                  </View>
                </Marker>
              </MapView>
            </View>
          )}
        </View>
      </ScrollView>

      {/* Simulate Crash Button */}
      {!crashDetected && (
        <TouchableOpacity
          onPress={handleCrash}
          style={styles.simulateCrashButton}
        >
          <Text style={styles.buttonText}>Simulate Crash</Text>
        </TouchableOpacity>
      )}

      {/* Emergency Response Modal */}
      <Modal
        visible={showResponseModal}
        animationType="slide"
        transparent={false}
        onRequestClose={() => setShowResponseModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Emergency Response</Text>
            <TouchableOpacity 
              onPress={() => setShowResponseModal(false)}
              style={styles.closeButton}
            >
              <Ionicons name="close" size={28} color="#FFF" />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.modalContent}>
            {/* Emergency Summary */}
            <View style={styles.responseSection}>
              <Text style={styles.responseSectionTitle}>
                <Ionicons name="alert-circle" size={20} color="#E74C3C" />{' '}
                Emergency Summary
              </Text>
              {emergencyResponse?.transcript_analysis?.summary && (
                <Text style={styles.responseText}>
                  {emergencyResponse.transcript_analysis.summary}
                </Text>
              )}
            </View>

            {/* Location Information */}
            <View style={styles.responseSection}>
              <Text style={styles.responseSectionTitle}>
                <Ionicons name="location" size={20} color="#3498DB" />{' '}
                Your Location
              </Text>
              {location && (
                <>
                  <Text style={styles.responseText}>
                    Latitude: {location.latitude.toFixed(6)}
                  </Text>
                  <Text style={styles.responseText}>
                    Longitude: {location.longitude.toFixed(6)}
                  </Text>
                  <TouchableOpacity
                    onPress={() => {
                      const mapUrl = `https://www.google.com/maps/search/?api=1&query=${location.latitude},${location.longitude}`;
                      Linking.openURL(mapUrl);
                    }}
                    style={styles.mapButton}
                  >
                    <Text style={styles.mapButtonText}>Open in Maps</Text>
                  </TouchableOpacity>
                </>
              )}
            </View>

            {/* Nearby Services */}
            <View style={styles.responseSection}>
              <Text style={styles.responseSectionTitle}>
                <Ionicons name="map" size={20} color="#2ECC71" />{' '}
                Nearby Emergency Services
              </Text>
              
              {nearbyServices && (
                <View style={styles.servicesMapContainer}>
                  <MapView
                    style={styles.servicesMap}
                    provider={PROVIDER_GOOGLE}
                    initialRegion={{
                      latitude: location.latitude,
                      longitude: location.longitude,
                      latitudeDelta: 0.05,
                      longitudeDelta: 0.05,
                    }}
                  >
                    {/* User location marker */}
                    <Marker
                      coordinate={{ latitude: location.latitude, longitude: location.longitude }}
                      title="Your Location"
                    >
                      <View style={[styles.markerContainer, { backgroundColor: '#E74C3C' }]}>
                        <Ionicons name="alert-circle" size={20} color="white" />
                      </View>
                    </Marker>

                    {/* Render service markers using the updated coordinates */}
                    {Object.entries(nearbyServices).map(([type, service]) => 
                      renderServiceMarker(service, type)
                    )}
{Object.entries(serviceRoutes).map(([type, route]) => (
    route && (
      <Polyline
        key={type}
        coordinates={decodePolyline(route).map(coord => ({
          latitude: coord[0],
          longitude: coord[1]
        }))}
        strokeColor={
          type === 'police' ? '#2E86C1' :
          type === 'hospital' ? '#E74C3C' :
          type === 'firebrigade' ? '#E67E22' : '#3498DB'
        }
        strokeWidth={3}
      />
    )
  ))}
  </MapView>

                  <View style={styles.servicesList}>
                    {Object.entries(nearbyServices).map(([type, service]) => (
                      <View key={type} style={styles.serviceItem}>
                        <FontAwesome 
                          name={type === 'police' ? 'shield' : type === 'hospital' ? 'hospital' : 'fire'} 
                          size={20} 
                          color={type === 'police' ? '#2E86C1' : type === 'hospital' ? '#E74C3C' : '#E67E22'} 
                        />
                        <Text style={styles.serviceText}>
                          {service.name} ({service.distance_meters.toFixed(0)}m) has been notified - 022-2554-1889
                        </Text>
                      </View>
                    ))}
                  </View>
                </View>
              )}
            </View>

            {/* Recommendations */}
            {emergencyResponse?.transcript_analysis?.suggestion && (
              <View style={styles.responseSection}>
                <Text style={styles.responseSectionTitle}>
                  <Ionicons name="medkit" size={20} color="#E74C3C" />{' '}
                  Recommendations
                </Text>
                <Text style={styles.responseText}>
                  {emergencyResponse.transcript_analysis.suggestion}
                </Text>
              </View>
            )}
          </ScrollView>

          <TouchableOpacity
            onPress={() => setShowResponseModal(false)}
            style={styles.modalCloseButton}
          >
            <Text style={styles.modalCloseButtonText}>Close</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F8F8',
    paddingTop: Platform.OS === 'android' ? 30 : 0,
  },
  scrollViewContent: {
    flexGrow: 1,
    alignItems: 'center',
    paddingBottom: 100, // To make space for the simulate crash button
  },
  card: {
    width: '95%',
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 20,
    marginTop: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 8,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#EEE',
    paddingBottom: 15,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#333',
    marginLeft: 10,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#ECF0F1',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  statusIndicator: {
    paddingVertical: 5,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  statusText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 14,
  },
  forceText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#555',
  },
  inputSection: {
    marginBottom: 25,
    paddingVertical: 15,
    borderTopWidth: 1,
    borderTopColor: '#EEE',
    paddingTop: 20,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#444',
    marginBottom: 15,
  },
  textInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F0F0F0',
    borderRadius: 10,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#DDD',
  },
  textInput: {
    flex: 1,
    padding: 15,
    fontSize: 16,
    color: '#333',
    minHeight: 60,
    textAlignVertical: 'top',
  },
  voiceButton: {
    padding: 15,
    backgroundColor: '#065F46',
    borderRadius: 10,
    marginLeft: 10,
    marginRight: 5,
  },
  recordingButton: {
    backgroundColor: '#065F46',
  },
  sendButton: {
    backgroundColor: '#065F46',
    padding: 18,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#065F46',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 6,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  messageContainer: {
    backgroundColor: '#ECF0F1',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  messageText: {
    fontSize: 16,
    color: '#555',
    textAlign: 'center',
  },
  transcriptBox: {
    marginTop: 10,
    backgroundColor: '#FFF',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#DEDEDE',
  },
  transcriptText: {
    fontSize: 15,
    color: '#333',
    fontStyle: 'italic',
  },
  alertBox: {
    backgroundColor: '#FADBD8',
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    borderWidth: 2,
    borderColor: '#E74C3C',
    shadowColor: '#E74C3C',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 10,
  },
  alertTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#E74C3C',
    marginBottom: 10,
    textAlign: 'center',
  },
  alertLocation: {
    fontSize: 16,
    color: '#E74C3C',
    marginBottom: 15,
    textAlign: 'center',
  },
  recordingContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 15,
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  recordingText: {
    fontSize: 18,
    color: '#065F46',
    fontWeight: 'bold',
    marginBottom: 10,
  },
  stopRecordingButton: {
    backgroundColor: '#065F46',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    width: '80%',
  },
  capturedImage: {
    width: 200,
    height: 150,
    resizeMode: 'cover',
    borderRadius: 10,
    marginTop: 15,
    borderWidth: 1,
    borderColor: '#DDD',
  },
  imagePreviewContainer: {
    alignItems: 'center',
    marginBottom: 15,
  },
  cancelButton: {
    backgroundColor: '#95A5A6',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 20,
    width: '100%',
  },
  section: {
    marginBottom: 25,
    width: '100%',
  },
  map: {
    width: '100%',
    height: 250,
    borderRadius: 10,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: '#DDD',
  },
  mapLinkButton: {
    backgroundColor: '#065F46',
    padding: 15,
    borderRadius: 10,
    alignSelf: 'center',
    width: '80%',
    alignItems: 'center',
    shadowColor: '#065F46',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 6,
  },
  mapLink: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  simulateCrashButton: {
    position: 'absolute',
    bottom: 20,
    backgroundColor: '#065F46',
    padding: 20,
    borderRadius: 30,
    alignItems: 'center',
    width: '90%',
    alignSelf: 'center',
    shadowColor: '#065F46',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 12,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#F8F8F8',
    paddingTop: Platform.OS === 'android' ? 30 : 0,
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: '#065F46',
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 8,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
  },
  closeButton: {
    padding: 5,
  },
  modalContent: {
    flex: 1,
    padding: 20,
  },
  responseSection: {
    backgroundColor: '#FFFFFF',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 5,
  },
  responseSectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
  },
  responseText: {
    fontSize: 16,
    color: '#555',
    lineHeight: 24,
  },
  mapButton: {
    backgroundColor: '#065F46',
    padding: 12,
    borderRadius: 8,
    marginTop: 15,
    alignItems: 'center',
  },
  mapButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  servicesMapContainer: {
    borderRadius: 10,
    overflow: 'hidden',
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#DDD',
  },
  servicesMap: {
    width: '100%',
    height: 200,
  },
  servicesList: {
    marginTop: 15,
  },
  serviceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F8F8F8',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#EEE',
  },
  serviceText: {
    fontSize: 15,
    color: '#333',
    marginLeft: 10,
    flexShrink: 1,
  },
  markerContainer: {
    padding: 8,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: '#FFF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
    elevation: 4,
  },
  modalCloseButton: {
    backgroundColor: '#065F46',
    padding: 18,
    alignItems: 'center',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 8,
  },
  modalCloseButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

export default App;