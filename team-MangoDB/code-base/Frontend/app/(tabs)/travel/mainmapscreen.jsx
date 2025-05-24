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
} from "react-native";
import MapView, { Polyline, Marker } from "react-native-maps";
import { useLocalSearchParams } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { MaterialIcons, FontAwesome, FontAwesome6 } from "@expo/vector-icons";
import Modal from "react-native-modal";
import { router } from "expo-router";

// Minimal map style for a cleaner look
const MINIMAL_MAP_STYLE = [
  { elementType: "geometry", stylers: [{ color: "#f5f5f5" }] },
  { elementType: "labels.icon", stylers: [{ visibility: "off" }] },
  { elementType: "labels.text.fill", stylers: [{ color: "#616161" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#f5f5f5" }] },
  {
    featureType: "poi",
    elementType: "geometry",
    stylers: [{ color: "#e8f5e9" }],
  },
  {
    featureType: "road",
    elementType: "geometry",
    stylers: [{ color: "#ffffff" }],
  },
  {
    featureType: "water",
    elementType: "geometry",
    stylers: [{ color: "#b3e5fc" }],
  },
];

// Decode polyline function
const decodePolyline = (encoded) => {
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
  const { id, route, overview_polyline } = useLocalSearchParams();
  console.log(route, overview_polyline);
  const [decodedLegs, setDecodedLegs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [selectedLegs, setSelectedLegs] = useState([]);
  const [amount, setAmount] = useState(0); // Track the total cost
  const mapRef = useRef(null);
  const [renderMap, setRenderMap] = useState(false); // State to track if map is rendered

  const modeColors = {
    WALK: "#77DD77", // Light green
    RAIL: "#779ECB", // Soft blue
    BUS: "#FFB347", // Orange
    METRO: "#D3A4FF", // Lavender
  };

  // Handle leg selection
  const handleLegSelection = (leg) => {
    if (leg.mode === "WALK") return; // Ignore selection for walking legs

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
        return (
          <MaterialIcons name="train" size={24} color={modeColors["RAIL"]} />
        );
      case "BUS":
        return <FontAwesome name="bus" size={24} color={modeColors["BUS"]} />;
      case "SUBWAY":
        return (
          <FontAwesome name="subway" size={24} color={modeColors["METRO"]} />
        );
      default:
        return <FontAwesome name="question" size={24} color="red" />;
    }
  };

  const [debouncedRoute, setDebouncedRoute] = useState(route);
  useEffect(() => {
    if (route) {
      setDebouncedRoute(route);
    } else if (overview_polyline) {
      setDebouncedRoute(overview_polyline);
    }
    return () => {
      setDebouncedRoute(null);
    };
  }, [route, overview_polyline]);

  const handleMapReady = () => {
    setTimeout(() => {
      if (decodedLegs.length > 0) {
        const firstLeg = decodedLegs[0].coordinates[0];
        mapRef.current.animateToRegion(
          {
            latitude: firstLeg.latitude,
            longitude: firstLeg.longitude,
            latitudeDelta: 0.02,
            longitudeDelta: 0.02,
          },
          1000
        );
      }
      setRenderMap(true);
    }, 200);
  };

  // Update decoded legs on screen focus
  useFocusEffect(
    React.useCallback(() => {
      if (debouncedRoute && !overview_polyline) {
        try {
          const parsedRoute = JSON.parse(debouncedRoute);
          setModalContent(parsedRoute);
          if (parsedRoute.legs) {
            const legsWithColors = parsedRoute.legs.map((leg) => {
              const coordinates = decodePolyline(leg.legGeometry.points);
              return {
                coordinates,
                color: modeColors[leg.mode] || "#000000",
                mode: leg.mode,
              };
            });
            setDecodedLegs(legsWithColors);
          } else {
            console.warn("Route legs are missing.");
          }
        } catch (error) {
          console.error("Error parsing route:", error);
        } finally {
          setIsLoading(false);
        }
      } else if (debouncedRoute && overview_polyline) {
        const coordinates = decodePolyline(debouncedRoute);
        setDecodedLegs([{ coordinates, color: "#000000" }]);
        setIsLoading(false);
      } else {
        setIsLoading(false);
      }
    }, [debouncedRoute])
  );

  useEffect(() => {
    return () => {
      setDecodedLegs([]);
      setModalContent(null);
      setSelectedLegs([]);
      console.log("Cleanup completed");
    };
  }, []);

  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text>Loading map data...</Text>
      </View>
    );
  }

  if (!debouncedRoute) {
    return (
      <View style={styles.errorContainer}>
        <Text>No route data available. Please go back and try again.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <MapView
        ref={mapRef}
        style={styles.map}
        customMapStyle={MINIMAL_MAP_STYLE}
        onMapReady={handleMapReady}
        region={{
          latitude:
            decodedLegs.length > 0
              ? decodedLegs[0].coordinates[0].latitude
              : 19.29462,
          longitude:
            decodedLegs.length > 0
              ? decodedLegs[0].coordinates[0].longitude
              : 72.85618,
          latitudeDelta: 0.1,
          longitudeDelta: 0.1,
        }}
        showsUserLocation={true}
      >
        {renderMap &&
          decodedLegs.map((leg, index) => (
            <React.Fragment key={index}>
              <Polyline
                coordinates={leg.coordinates}
                strokeColor={leg.color}
                strokeWidth={4}
              />
              <Marker
                coordinate={leg.coordinates[0]}
                pinColor={index === 0 ? "green" : "black"}
                onPress={() => {
                  Alert.alert(
                    "Street View",
                    "Would you like to view this location in Street View?",
                    [
                      { text: "Cancel", style: "cancel" },
                      {
                        text: "View",
                        onPress: () =>
                          router.push({
                            pathname: "/(tabs)/travel/PhotoView",
                            params: {
                              latitude: leg.coordinates[0].latitude,
                              longitude: leg.coordinates[0].longitude,
                            },
                          }),
                      },
                    ]
                  );
                }}
              />
              <Marker
                coordinate={leg.coordinates[leg.coordinates.length - 1]}
                pinColor={index === decodedLegs.length - 1 ? "red" : "black"}
                onPress={() => {
                  Alert.alert(
                    "Street View",
                    "Would you like to view this location in Street View?",
                    [
                      { text: "Cancel", style: "cancel" },
                      {
                        text: "View",
                        onPress: () =>
                          router.push({
                            pathname: "/(tabs)/travel/PhotoView",
                            params: {
                              latitude:
                                leg.coordinates[leg.coordinates.length - 1]
                                  .latitude,
                              longitude:
                                leg.coordinates[leg.coordinates.length - 1]
                                  .longitude,
                            },
                          }),
                      },
                    ]
                  );
                }}
              />
            </React.Fragment>
          ))}
      </MapView>

      <TouchableHighlight
        underlayColor="rgba(6, 95, 70, 1)"
        style={[styles.button, { bottom: 80 }]}
        onPress={() => {
          router.push("/(tabs)/travel/PhotoView");
        }}
      >
        <Text style={styles.buttonText}>
          <FontAwesome6 name="street-view" size={24} color="white" />
        </Text>
      </TouchableHighlight>
      <TouchableHighlight
        underlayColor="rgba(6, 95, 70, 1)"
        style={[styles.button]}
        onPress={() => {
          setIsModalVisible(true);
        }}
      >
        <Text style={styles.buttonText}>View Route</Text>
      </TouchableHighlight>

      <Modal
        isVisible={isModalVisible}
        onBackdropPress={() => setIsModalVisible(false)}
        onBackButtonPress={() => setIsModalVisible(false)}
      >
        <View style={styles.modalContent}>
          {modalContent ? (
            <View style={styles.modalInnerContainer}>
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
                        {leg.mode === "WALK" ? "Walk" : leg.route} ·
                        {leg.distance > 1000
                          ? ` ${Math.round(leg.distance / 1000)} km`
                          : ` ${Math.round(leg.distance)} m`}
                      </Text>
                      <Text style={styles.legSubInfo}>
                        {leg.from.name} → {leg.to.name}
                      </Text>
                      <Text style={styles.legSubInfo}>
                        {new Date(leg.startTime).toLocaleTimeString()} -{" "}
                        {new Date(leg.endTime).toLocaleTimeString()}
                      </Text>
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
                  underlayColor="rgba(6, 95, 70, 0.98)"
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
                      ? "Select rides to book"
                      : `Book ${selectedLegs.length} selected ${
                          selectedLegs.length === 1 ? "ride" : "rides"
                        }`}
                  </Text>
                </TouchableHighlight>
              </View>
            </View>
          ) : (
            <Text>No route details available.</Text>
          )}
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  map: {
    flex: 1,
    ...StyleSheet.absoluteFillObject,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  errorContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  container: {
    flex: 1,
    backgroundColor: "#1a1a1a",
  },
  legsContainer: {
    flex: 1,
  },
  leg: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 16,
    padding: 10,
    borderRadius: 8,
  },
  selectedLeg: {
    backgroundColor: "#f0fdf4",
  },
  iconContainer: {
    width: 20,
    alignItems: "center",
    marginRight: 20,
  },
  verticalLine: {
    width: 2,
    height: 40,
    backgroundColor: "gray",
    marginTop: 8,
  },
  legDetails: {
    flex: 1,
    marginRight: 10,
  },
  legDescription: {
    fontSize: 16,
    color: "#000000",
    fontWeight: "bold",
  },
  legSubInfo: {
    fontSize: 14,
    color: "#666",
    marginTop: 4,
    fontWeight: "500",
  },
  button: {
    position: "absolute",
    bottom: 20,
    right: 20,
    backgroundColor: "rgba(6, 95, 70, 0.8)",
    padding: 15,
    borderRadius: 30,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    color: "white",
    fontWeight: "600",
  },
  modalContent: {
    backgroundColor: "white",
    padding: 20,
    borderRadius: 10,
    height: "80%",
  },
  modalInnerContainer: {
    flex: 1,
    flexDirection: "column",
  },
  checkboxContainer: {
    justifyContent: "center",
    alignItems: "center",
    paddingLeft: 10,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 4,
    borderWidth: 2,
    borderColor: "#064e3b",
    justifyContent: "center",
    alignItems: "center",
  },
  checkboxSelected: {
    backgroundColor: "#064e3b",
  },
  buttonContainer: {
    paddingTop: 10,
    paddingBottom: 10,
    backgroundColor: "white",
  },
  bookNowButton: {
    padding: 15,
    borderRadius: 5,
    alignItems: "center",
    width: "100%",
    backgroundColor: "#064e3b",
  },
  bookNowButtonDisabled: {
    backgroundColor: "#94a3b8",
  },
  bookNowButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "bold",
  },
  walkingLeg: {
    backgroundColor: "#ffffff",
  },
  checkboxDisabled: {
    width: 20,
    height: 20,
    borderRadius: 4,
    borderWidth: 2,
    borderColor: "#94a3b8",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#ffffff",
  },
});

export default MapScreen;
