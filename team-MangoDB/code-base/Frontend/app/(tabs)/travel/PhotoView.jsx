import { useLocalSearchParams } from "expo-router";
import React, { useEffect, useState } from "react";
import { View, StyleSheet, Text } from "react-native";
import { WebView } from "react-native-webview";

const PhotoView = () => {
  const { latitude, longitude } = useLocalSearchParams();
  const [heading, setHeading] = useState(0);

  useEffect(() => {
    console.log("Latitude:", latitude, "Longitude:", longitude);
  }, [latitude, longitude]);

  const streetViewHtml = `
  <!DOCTYPE html>
  <html>
    <head>
      <title>Street View</title>
      <script src="https://maps.googleapis.com/maps/api/js?key=${process.env.EXPO_PUBLIC_google_api}"></script>
      <script>
        function initialize() {
          try {
            const fenway = { lat: ${latitude}, lng: ${longitude} };
            const panorama = new google.maps.StreetViewPanorama(
              document.getElementById('street-view'),
              {
                position: fenway,
                pov: { heading: 100, pitch: 0 },
                zoom: 1,
              }
            );
          } catch (error) {
            document.body.innerHTML = "<h1>Failed to load Street View: " + error.message + "</h1>";
          }
        }
      </script>
    </head>
    <body onload="initialize()" style="margin:0; padding:0;">
      <div id="street-view" style="width: 100%; height: 100vh;"></div>
    </body>
  </html>
`;

  return (
    <View style={styles.container}>
      <WebView
        originWhitelist={["*"]}
        source={{ html: streetViewHtml }}
        style={styles.webview}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        onError={(syntheticEvent) => {
          const { nativeEvent } = syntheticEvent;
          console.error("WebView error: ", nativeEvent);
        }}
        onMessage={(event) => {
          console.log("WebView message: ", event.nativeEvent.data);
        }}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  webview: {
    height: 20,
    flex: 1,
    backgroundColor: "white",
  },
  text: {
    textAlign: "center",
    margin: 10,
    fontSize: 16,
  },
});

export default PhotoView;
