import React, { useEffect, useState, useRef } from "react";
import {
  ScrollView,
  StyleSheet,
  View,
  Text,
  ActivityIndicator,
  TouchableHighlight,
  TouchableOpacity,
  Alert,
  Dimensions,
  Linking,
  BackHandler,
} from "react-native";
import MapView, {
  Polyline,
  Marker,
  PROVIDER_DEFAULT,
  UrlTile,
} from "react-native-maps";
import { useLocalSearchParams, useRouter } from "expo-router"; // useRouter for navigation
import { useFocusEffect } from "@react-navigation/native";
import * as Location from "expo-location";
import { MaterialIcons, FontAwesome } from "@expo/vector-icons";
import Modal from "react-native-modal";
// Removed duplicate import of router, useRouter from expo-router is preferred

const { width, height } = Dimensions.get("window");

// Decode polyline function
const decodePolyline = (encoded) => {
  if (!encoded) return []; // Add null check for encoded string
  let index = 0,
    len = encoded.length;
  let lat = 0,
    lng = 0;
  const coordinates = [];

  while (index < len) {
    let b,
      shift = 0,
      result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    const dlat = result & 1 ? ~(result >> 1) : result >> 1;
    lat += dlat;

    shift = 0;
    result = 0;
    do {
      b = encoded.charCodeAt(index++) - 63;
      result |= (b & 0x1f) << shift;
      shift += 5;
    } while (b >= 0x20);
    const dlng = result & 1 ? ~(result >> 1) : result >> 1;
    lng += dlng;

    coordinates.push({
      latitude: lat / 1e5,
      longitude: lng / 1e5,
    });
  }
  return coordinates;
};

const MapScreen = () => {
  const params = useLocalSearchParams();
  const { id, route, overview_polyline } = params || {}; // Add null check for params
  const router = useRouter(); // Use useRouter hook for navigation

  const [decodedLegs, setDecodedLegs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [userLocation, setUserLocation] = useState(null);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [selectedLegs, setSelectedLegs] = useState([]);
  const [amount, setAmount] = useState(0);
  const mapRef = useRef(null);

  const modeColors = {
    WALK: "#77DD77",
    RAIL: "#779ECB",
    BUS: "#FFB347",
    SUBWAY: "#D3A4FF",
  };

  // Handle Back Button
  useEffect(() => {
    const backAction = () => {
      if (isModalVisible) {
        setIsModalVisible(false);
        return true; // Prevent default behavior (exit app)
      } 
      // If no modal, let router handle back navigation or default behavior
      // Check if router can go back, otherwise allow default (exit app or previous native screen)
      if (router.canGoBack()) {
        router.back();
        return true; // Prevent default behavior
      }
      return false; // Allow default behavior (exit app)
    };

    const backHandler = BackHandler.addEventListener(
      "hardwareBackPress",
      backAction
    );

    return () => backHandler.remove(); // Correctly remove listener
  }, [isModalVisible, router]);

  const handleLegSelection = (leg) => {
    if (leg.mode === "WALK") return;

    setSelectedLegs((prev) => {
      const isSelected = prev.some(
        (selectedLeg) =>
          selectedLeg.startTime === leg.startTime &&
          selectedLeg.endTime === leg.endTime
      );

      setAmount(modalContent?.totalCost || 0);

      if (isSelected) {
        return prev.filter(
          (selectedLeg) =>
            selectedLeg.startTime !== leg.startTime ||
            selectedLeg.endTime !== leg.endTime
        );
      } else {
        return [...prev, leg];
      }
    });
  };

  const isLegSelected = (leg) => {
    if (leg.mode === "WALK") return false;
    return selectedLegs.some(
      (selectedLeg) =>
        selectedLeg.startTime === leg.startTime &&
        selectedLeg.endTime === leg.endTime
    );
  };

  const renderLegIcon = (mode) => {
    switch (mode) {
      case "WALK":
        return <FontAwesome name="male" size={24} color={modeColors["WALK"]} />;
      case "RAIL":
        return <MaterialIcons name="train" size={24} color={modeColors["RAIL"]} />;
      case "BUS":
        return <FontAwesome name="bus" size={24} color={modeColors["BUS"]} />;
      case "SUBWAY":
        return <FontAwesome name="subway" size={24} color={modeColors["SUBWAY"]} />;
      default:
        return <FontAwesome name="question" size={24} color="red" />;
    }
  };

  const openGoogleMaps = (fromLat, fromLon, toLat, toLon) => {
    const url = `https://www.google.com/maps/dir/?api=1&origin=${fromLat},${fromLon}&destination=${toLat},${toLon}`;
    Linking.openURL(url).catch((err) => console.error("Error opening Google Maps:", err));
  };

  const [debouncedRoute, setDebouncedRoute] = useState(route);
  useEffect(() => {
    let handler;
    if (route) {
      handler = setTimeout(() => setDebouncedRoute(route), 300);
    } else if (overview_polyline) {
      setDebouncedRoute(overview_polyline);
    }
    return () => clearTimeout(handler);
  }, [route, overview_polyline]);

  useEffect(() => {
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        console.error("Permission to access location was denied");
        Alert.alert("Permission Denied", "Location permission is needed to show your current location on the map.");
        return;
      }
      try {
        const location = await Location.getCurrentPositionAsync({});
        setUserLocation(location.coords);
      } catch (error) {
        console.error("Error getting current position:", error);
        Alert.alert("Location Error", "Could not fetch your current location.");
      }
    })();
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      setIsLoading(true); // Reset loading state on focus
      if (debouncedRoute && !overview_polyline) {
        try {
          const parsedRoute = JSON.parse(debouncedRoute);
          setModalContent(parsedRoute);
          if (parsedRoute.legs && Array.isArray(parsedRoute.legs)) {
            const legsWithColors = parsedRoute.legs.map((leg) => {
              const coordinates = decodePolyline(leg.legGeometry?.points); // Optional chaining for points
              return {
                coordinates,
                color: modeColors[leg.mode] || "#000000",
                mode: leg.mode,
              };
            });
            setDecodedLegs(legsWithColors);
          } else {
            setDecodedLegs([]); // Ensure decodedLegs is an array
          }
        } catch (error) {
          console.error("Error parsing route:", error);
          setDecodedLegs([]); // Set to empty array on error
        } finally {
          setIsLoading(false);
        }
      } else if (debouncedRoute && overview_polyline) {
        const coordinates = decodePolyline(debouncedRoute);
        setDecodedLegs([{ coordinates, color: "#000000" }]);
        setIsLoading(false);
      } else {
        console.log("No route or overview_polyline available");
        setDecodedLegs([]); // Ensure decodedLegs is an array
        setIsLoading(false);
      }
    }, [debouncedRoute, overview_polyline])
  );

  useEffect(() => {
    if (decodedLegs.length > 0 && mapRef.current && decodedLegs.every(leg => leg.coordinates.length > 0)) {
      const allCoordinates = decodedLegs.flatMap((leg) => leg.coordinates);
      if (allCoordinates.length === 0) return;

      const latitudes = allCoordinates.map((point) => point.latitude);
      const longitudes = allCoordinates.map((point) => point.longitude);

      const minLat = Math.min(...latitudes);
      const maxLat = Math.max(...latitudes);
      const minLng = Math.min(...longitudes);
      const maxLng = Math.max(...longitudes);

      if (isFinite(minLat) && isFinite(maxLat) && isFinite(minLng) && isFinite(maxLng)) {
         mapRef.current.animateToRegion({
          latitude: (minLat + maxLat) / 2,
          longitude: (minLng + maxLng) / 2,
          latitudeDelta: Math.max(0.01, maxLat - minLat + 0.01),
          longitudeDelta: Math.max(0.01, maxLng - minLng + 0.01),
        });
      }
    }
  }, [decodedLegs]);

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#047857" />
        <Text style={styles.loadingText}>Loading map data...</Text>
      </View>
    );
  }

  if (!debouncedRoute && !overview_polyline) {
    return (
      <View style={styles.errorContainer}>
        <Text>No route data available. Please go back and try again.</Text>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }
  
  if (decodedLegs.length === 0 && !isLoading) {
     return (
      <View style={styles.errorContainer}>
        <Text>Could not decode route. Please try again.</Text>
         <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        provider={PROVIDER_DEFAULT}
        initialRegion={
          decodedLegs.length > 0 && decodedLegs[0].coordinates.length > 0
            ? {
                latitude: decodedLegs[0].coordinates[0].latitude,
                longitude: decodedLegs[0].coordinates[0].longitude,
                latitudeDelta: 0.02,
                longitudeDelta: 0.02,
              }
            : userLocation 
            ? { 
                latitude: userLocation.latitude,
                longitude: userLocation.longitude,
                latitudeDelta: 0.02,
                longitudeDelta: 0.02,
             }
            : {
                latitude: 19.0760, // Default to Mumbai if no data and no user location
                longitude: 72.8777,
                latitudeDelta: 0.1,
                longitudeDelta: 0.1,
              }
        }
        showsUserLocation={true}
        zoomEnabled={true}
        scrollEnabled={true}
      >
        <UrlTile
          urlTemplate="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maximumZ={19}
        />

        {decodedLegs.map((leg, index) => (
          <React.Fragment key={index}>
            {leg.coordinates && leg.coordinates.length > 0 && (
              <Polyline
                coordinates={leg.coordinates}
                strokeColor={leg.color}
                strokeWidth={4}
              />
            )}
            {leg.coordinates && leg.coordinates.length > 0 && (
              <Marker
                coordinate={leg.coordinates[0]}
                title="Start of Leg"
                pinColor={index === 0 ? "green" : leg.color || "black"}
              />
            )}
            {leg.coordinates && leg.coordinates.length > 0 && (
              <Marker
                coordinate={leg.coordinates[leg.coordinates.length - 1]}
                title="End of Leg"
                pinColor={index === decodedLegs.length - 1 ? "red" : leg.color || "black"}
              />
            )}
          </React.Fragment>
        ))}
      </MapView>

      <View style={styles.legendContainer}>
        {Object.entries(modeColors).map(([mode, color]) => (
          <View key={mode} style={styles.legendItem}>
            <View style={[styles.legendColor, { backgroundColor: color }]} />
            <Text style={styles.legendText}>{mode.toUpperCase()}</Text>
          </View>
        ))}
      </View>

      <TouchableHighlight
        underlayColor="#047857" // Updated color
        style={styles.floatingButton}
        onPress={() => setIsModalVisible(true)}
      >
        <MaterialIcons name="directions" size={24} color="white" />
      </TouchableHighlight>

      <Modal
        isVisible={isModalVisible}
        onBackdropPress={() => setIsModalVisible(false)}
        onBackButtonPress={() => {
            setIsModalVisible(false);
            return true; // Consume the event
        }}
        style={styles.modal}
        useNativeDriverForBackdrop
      >
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Journey Details</Text>
            <TouchableOpacity onPress={() => setIsModalVisible(false)}>
              <MaterialIcons name="close" size={24} color="#334155" />
            </TouchableOpacity>
          </View>
          
          {modalContent && modalContent.legs ? (
            <View style={styles.modalInnerContainer}>
              <View style={styles.routeSummary}>
                <View style={styles.summaryItem}>
                  <MaterialIcons name="access-time" size={18} color="#047857" />
                  <Text style={styles.summaryText}>
                    {Math.round(modalContent.duration / 60)} min
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <MaterialIcons name="straighten" size={18} color="#047857" />
                  <Text style={styles.summaryText}>
                    {(() => {
                      const totalDistance = modalContent.legs.reduce((sum, leg) => sum + (leg.distance || 0), 0);
                      return totalDistance > 1000
                        ? `${(totalDistance / 1000).toFixed(1)} km`
                        : `${Math.round(totalDistance)} m`;
                    })()}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <MaterialIcons name="payments" size={18} color="#047857" />
                  <Text style={styles.summaryText}>
                    ₹{modalContent.totalCost?.toFixed(2) || "0.00"}
                  </Text>
                </View>
              </View>
              
              <ScrollView style={styles.legsContainer}>
                {modalContent.legs.map((leg, index) => (
                  <TouchableOpacity
                    key={index}
                    style={[
                      styles.leg,
                      isLegSelected(leg) && styles.selectedLeg,
                      leg.mode === "WALK" && styles.walkingLeg,
                    ]}
                    onPress={() => handleLegSelection(leg)}
                    disabled={leg.mode === "WALK"}
                  >
                    <View style={styles.iconContainer}>
                      {renderLegIcon(leg.mode)}
                      {index !== modalContent.legs.length - 1 && (
                        <View style={styles.verticalLine} />
                      )}
                    </View>
                    <View style={styles.legDetails}>
                      <Text
                        style={[
                          styles.legDescription,
                          leg.mode === "WALK" && styles.walkingLegText,
                        ]}
                      >
                        {leg.mode === "WALK"
                          ? "Walk"
                          : leg.mode === "BUS"
                          ? `Bus ${leg.routeShortName || ''} · ${(leg.route || '').toUpperCase()}`
                          : `${(leg.route || '').toUpperCase()}`}{" "}
                        ·{" "}
                        {leg.distance > 1000
                          ? `${(leg.distance / 1000).toFixed(1)} km`
                          : `${Math.round(leg.distance || 0)} m`}
                      </Text>
                      <Text style={styles.legSubInfo}>
                        {(leg.from?.name || 'N/A').toUpperCase()} → {(leg.to?.name || 'N/A').toUpperCase()}
                      </Text>
                      <Text style={styles.legSubInfo}>
                        {new Date(leg.startTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} -{" "}
                        {new Date(leg.endTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </Text>
                      {leg.from?.lat && leg.from?.lon && leg.to?.lat && leg.to?.lon && (
                        <TouchableOpacity
                            style={styles.directionsButton}
                            onPress={() => openGoogleMaps(leg.from.lat, leg.from.lon, leg.to.lat, leg.to.lon)}
                        >
                            <Text style={styles.directionsButtonText}>Get Directions</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                    <View style={styles.checkboxContainer}>
                      {leg.mode !== "WALK" ? (
                        <View
                          style={[
                            styles.checkbox,
                            isLegSelected(leg) && styles.checkboxSelected,
                          ]}
                        >
                          {isLegSelected(leg) && (
                            <MaterialIcons
                              name="check"
                              size={16}
                              color="white"
                            />
                          )}
                        </View>
                      ) : (
                        <View style={styles.checkboxDisabled}>
                          <MaterialIcons
                            name="block"
                            size={16}
                            color="#94a3b8"
                          />
                        </View>
                      )}
                    </View>
                  </TouchableOpacity>
                ))}
              </ScrollView>
              <View style={styles.buttonContainer}>
                <TouchableHighlight
                  underlayColor="#056040" // Darker shade for press
                  style={[
                    styles.bookNowButton,
                    selectedLegs.length === 0 && styles.bookNowButtonDisabled,
                  ]}
                  disabled={selectedLegs.length === 0}
                  onPress={() => {
                    setIsModalVisible(false);
                    router.push({
                      pathname: "/(tabs)/travel/booking",
                      params: {
                        selectedLegs: JSON.stringify(selectedLegs),
                        amount: amount,
                      },
                    });
                  }}
                >
                  <Text style={styles.bookNowButtonText}>
                    {selectedLegs.length === 0
                      ? "Select Transport Modes to Book"
                      : `Book Now for ₹${amount.toFixed(2)}`}
                  </Text>
                </TouchableHighlight>
              </View>
            </View>
          ) : (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#047857" />
              <Text style={styles.loadingText}>Loading route details...</Text>
            </View>
          )}
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f8fafc", // Lighter background
  },
  map: {
    width: width,
    height: height,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#f8fafc",
  },
  loadingText: {
    marginTop: 12,
    color: "#475569", // Softer text color
    fontSize: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#f8fafc",
  },
  backButton: {
    marginTop: 20,
    backgroundColor: "#047857",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  backButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "500",
  },
  floatingButton: {
    backgroundColor: "#047857", // Updated color
    width: 56,
    height: 56,
    borderRadius: 28,
    position: "absolute",
    bottom: 30, // Adjusted position
    right: 20,
    alignItems: "center",
    justifyContent: "center",
    elevation: 6, // Slightly more elevation
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  modal: {
    margin: 0,
    justifyContent: "flex-end",
  },
  modalContent: {
    backgroundColor: "white",
    borderTopLeftRadius: 20, // Smoother radius
    borderTopRightRadius: 20,
    paddingBottom: 20,
    maxHeight: "90%", // Adjusted max height
    paddingTop: 8, // Add some top padding
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20, // Increased padding
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#e2e8f0",
  },
  modalTitle: {
    fontSize: 20, // Larger title
    fontWeight: "600",
    color: "#1e293b", // Darker title
  },
  modalInnerContainer: {
    paddingHorizontal: 20, // Consistent padding
  },
  routeSummary: {
    flexDirection: "row",
    justifyContent: "space-around",
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#e2e8f0",
    marginBottom: 8, // Add some margin below summary
  },
  summaryItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  summaryText: {
    marginLeft: 8,
    fontSize: 15, // Slightly smaller summary text
    fontWeight: "500",
    color: "#334155",
  },
  legsContainer: {
    maxHeight: height * 0.5, // Max height relative to screen
  },
  leg: {
    flexDirection: "row",
    paddingVertical: 16, // More vertical padding
    paddingHorizontal: 8, 
    borderBottomWidth: 1,
    borderBottomColor: "#f1f5f9", // Lighter border
    alignItems: "center",
  },
  selectedLeg: {
    backgroundColor: "#e0f2fe", // Lighter blue for selection
    borderRadius: 8,
  },
  walkingLeg: {
    // No specific style needed if default is fine
  },
  walkingLegText: {
    fontStyle: "italic",
    color: "#64748b",
  },
  iconContainer: {
    marginRight: 16,
    alignItems: 'center',
    width: 30, 
  },
  verticalLine: {
    height: 35, // Taller line
    width: 2,
    backgroundColor: '#e2e8f0', // Lighter line
    marginVertical: 6,
    alignSelf: 'center',
  },
  legDetails: {
    flex: 1,
  },
  legDescription: {
    fontSize: 15,
    fontWeight: "600", // Bolder description
    color: "#1e293b",
    marginBottom: 4,
  },
  legSubInfo: {
    fontSize: 13,
    color: "#475569",
    marginBottom: 3,
    lineHeight: 18,
  },
  checkboxContainer: {
    marginLeft: 12,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 2,
    borderColor: "#94a3b8",
    backgroundColor: "transparent",
    justifyContent: "center",
    alignItems: "center",
  },
  checkboxSelected: {
    backgroundColor: "#047857", // Updated color
    borderColor: "#047857",
  },
  checkboxDisabled: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: "#e2e8f0", // Lighter disabled bg
    justifyContent: "center",
    alignItems: "center",
  },
  buttonContainer: {
    alignItems: "center",
    paddingVertical: 20, // More padding
    borderTopWidth: 1,
    borderTopColor: "#e2e8f0",
    marginTop: 8,
  },
  bookNowButton: {
    backgroundColor: "#047857", // Updated color
    paddingVertical: 14, // Larger button
    paddingHorizontal: 30,
    borderRadius: 10, // Smoother radius
    width: "90%",
    elevation: 2,
  },
  bookNowButtonDisabled: {
    backgroundColor: "#94a3b8",
    elevation: 0,
  },
  bookNowButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
    textAlign: "center",
  },
  legendContainer: {
    position: 'absolute',
    bottom: 20, // Adjusted for floating button
    left: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.95)', // More opaque
    padding: 12,
    borderRadius: 10,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 5,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 5,
  },
  legendColor: {
    width: 18, // Larger color swatch
    height: 18,
    borderRadius: 9,
    marginRight: 10,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  legendText: {
    fontSize: 14,
    color: '#334155',
    fontWeight: '500',
  },
  directionsButton: {
    marginTop: 10,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#047857',
    alignSelf: 'flex-start',
  },
  directionsButtonText: {
    color: '#047857',
    fontSize: 13,
    fontWeight: '600',
  },
});

export default MapScreen;