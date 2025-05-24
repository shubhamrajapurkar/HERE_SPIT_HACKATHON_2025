import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  FlatList,
  TouchableHighlight,
  TouchableOpacity,
  Dimensions,
  Animated,
  ActivityIndicator,
  Platform,
  ScrollView,
  Modal,
  Pressable,
  TextInput,
} from "react-native";
import { router } from "expo-router";
import { FontAwesome5, MaterialIcons } from "@expo/vector-icons";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { Audio } from "expo-av";
import { FadeIn, FadeInDown, FadeInUp } from "react-native-reanimated";

const { width, height } = Dimensions.get("window");

const Home = () => {
  const pastLocations = [
    { id: "1", name: "Sardar Patel Institute of Technology", address: "Andheri West" },
    { id: "2", name: "Jio World Centre", address: "Bandra Kurla Complex" },
    { id: "3", name: "Teen Hath Naka", address: "Thane West" },
  ];

  const rideOptions = [
    { id: "1", name: "Cab", icon: "car", gradient: ["#FFF", "#FFF"] },
    { id: "2", name: "Bus", icon: "bus", gradient: ["#FFF", "#FFF"] },
    { id: "3", name: "Train", icon: "train", gradient: ["#FFF", "#FFF"] },
    { id: "4", name: "Metro", icon: "subway", gradient: ["#FFF", "#FFF"] },
  ];

  const carouselData = [
    {
      id: "1",
      title: "Rides Completed",
      description: "Congratulations! You've completed 45 rides this month!",
      icon: "car",
      buttonText: "View Rides",
      gradient: ["#EAF7F2", "#D5F1E4"],
    },
    {
      id: "2",
      title: "Carbon Emission Saved",
      description: "You have saved 25kg of CO2 emissions so far!",
      icon: "leaf",
      buttonText: "See More",
      gradient: ["#F7FCEB", "#EEF8D8"],
    },
    {
      id: "3",
      title: "Eco-Friendly Tips",
      description: "Discover tips to reduce your carbon footprint.",
      icon: "lightbulb",
      buttonText: "Learn More",
      gradient: ["#EAF7F2", "#D5F1E4"],
    },
  ];

  const carouselRef = useRef(null);

  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState("idle");
  const [messages, setMessages] = useState([]);
  const [showRecordingModal, setShowRecordingModal] = useState(false);
  const [textInputContent, setTextInputContent] = useState("");

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const waveAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (transcript) {
      setTextInputContent(transcript);
    }
  }, [transcript]);

  const addMessage = (text, type = "info") => {
    setMessages((prev) => [
      ...prev,
      { text, type, timestamp: new Date().toISOString() },
    ]);
  };

  const startRecording = async () => {
    try {
      if (recording) {
        await recording.stopAndUnloadAsync();
        setRecording(null);
        setIsRecording(false);
      }

      setTranscript("");
      setTextInputContent("");
      setMessages([]);
      setShowRecordingModal(true);

      setStatus("requesting permissions");
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      setStatus("starting recording");
      const { recording: newRecording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      setRecording(newRecording);
      setIsRecording(true);
      setStatus("recording");
      addMessage('Recording started...', 'info');
      startPulseAnimation();
      startWaveAnimation();
    } catch (err) {
      console.error("Failed to start recording", err);
      setStatus("error");
      addMessage('Failed to start recording: ' + err.message, 'error');
    }
  };

  const stopRecording = async () => {
    setStatus("stopping recording");
    if (!recording) {
      addMessage("No recording to stop.", "warning");
      return;
    }
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);
    setIsRecording(false);
    pulseAnim.stopAnimation();
    waveAnim.stopAnimation();

    transcribeAudio(uri);
  };

  const transcribeAudio = async (audioUri) => {
    try {
      setStatus("transcribing");
      addMessage('Transcribing audio...', 'info');

      const formData = new FormData();
      const fileUri =
        Platform.OS === "android" ? audioUri : audioUri.replace("file://", "");

      formData.append("file", {
        uri: fileUri,
        type: "audio/m4a",
        name: "recording.m4a",
      });

      const backendUrl = process.env.EXPO_PUBLIC_BACKEND_IP;
      const url = `${backendUrl}/transcribe`;
      console.log("Sending transcription request to:", url);

      const response = await fetch(url, {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Server response:", errorText);
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      if (!result.text) {
        throw new Error("Invalid response format");
      }

      setTranscript(result.text);
      setStatus("transcribed");
      addMessage('Transcription completed!', 'success');
    } catch (error) {
      console.error("Transcription error:", error);
      setStatus("error");
      addMessage(`Transcription failed: ${error.message}`, "error");
    } finally {
      // Keep modal open after transcription for user interaction
    }
  };

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.08,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const startWaveAnimation = () => {
    Animated.loop(
      Animated.timing(waveAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
      })
    ).start();
  };

  const handleSendMessage = async () => {
    if (textInputContent.trim()) {
      try {
        const response = await fetch(`http://${process.env.EXPO_PUBLIC_machine_ip}:5001/route`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            prompt: textInputContent
          })
        });
        
        const data = await response.json();
        console.log("Transcript:", transcript);
        console.log('Response:', data);
        
        if (data.type == 'car') {
          router.push({
            pathname: '/travel/testmapscreen',
            params: { resp_data: JSON.stringify(data) }
          });
        } else {
          router.push({
            pathname: '/travel/routescreen1',
            params: { resp_data: JSON.stringify(data) }
          });
        }
        
        addMessage(`User: ${textInputContent}`, 'sent');
        setTextInputContent("");
        setTranscript("");
        setStatus("idle");
      } catch (error) {
        console.error('Error sending request:', error);
        addMessage("Failed to send message.", "error");
      }
    } else {
      addMessage("Message cannot be empty.", "warning");
    }
  };

  return (
    <SafeAreaView className="min-h-full bg-white">
      <LinearGradient colors={["#FFF", "#FFFFFF"]} className="flex-1 px-4 pt-8">
        <FlatList
          data={[1, 1, 1, 1, 1, 1]}
          renderItem={({ item, index }) => {
            return (
              <View className="w-full">
                {index === 0 && (
                  <View className="mx-4 mt-6 flex-row justify-between items-center">
                    <Text className="text-4xl font-pbold bg-clip-text text-black">Hi Nimit,</Text>
                    <View className="w-16 h-16" />
                  </View>
                )}

                {index === 1 && (
                  <View className="relative">
                    <TouchableHighlight
                      className="mt-4 overflow-hidden rounded-full"
                      underlayColor="#e5e7eb"
                      onPress={() => router.push("/travel")}
                    >
                      <View className="flex-row items-center px-4 py-4 bg-gray-100 rounded-full">
                        <FontAwesome5
                          name="search"
                          size={18}
                          color="#6B7280"
                          className="mr-3 ml-1"
                        />
                        <Text className="text-lg text-gray-800 font-pmedium my-[1px]">Where to?</Text>
                      </View>
                    </TouchableHighlight>
                    
                    <TouchableOpacity
                      onPress={() => {
                        setTranscript("");
                        setTextInputContent("");
                        setMessages([]);
                        setStatus("idle");
                        setShowRecordingModal(true);
                      }}
                      className="absolute right-2 top-6 w-12 h-12 rounded-full items-center justify-center bg-gray-200 shadow-lg"
                      style={{
                        zIndex: 10,
                        elevation: 10,
                        shadowColor: "#6B7280",
                        shadowOffset: { width: 0, height: 2 },
                        shadowOpacity: 0.2,
                        shadowRadius: 4,
                        borderWidth: 1,
                        borderColor: 'rgba(0,0,0,0.05)'
                      }}
                      activeOpacity={0.8}
                    >
                      <FontAwesome5 name="microphone" size={20} color="#4B5563" />
                    </TouchableOpacity>
                  </View>
                )}

                {index === 2 && (
                  <FlatList
                    className="mt-4"
                    data={pastLocations.slice(0, 3)}
                    keyExtractor={(item) => item.id}
                    renderItem={({ item }) => (
                      <TouchableHighlight
                        underlayColor="#d1d5db"
                        onPress={() => { }}
                        className="rounded-lg overflow-hidden mb-2"
                      >
                        <View className="flex-row items-center bg-white p-3 rounded-lg shadow-md shadow-gray-400">
                          <MaterialIcons name="location-on" size={28} color="#6b7280" />
                          <View className="ml-3">
                            <Text className="font-psemibold text-lg text-gray-800">{item.name}</Text>
                            <Text className="font-pregular text-sm text-gray-500">
                              {item.address}
                            </Text>
                          </View>
                        </View>
                      </TouchableHighlight>
                    )}
                    ItemSeparatorComponent={() => <View className="h-[1px] bg-gray-200 mx-3" />}
                  />
                )}

                {index === 3 && (
                  <View className="mt-6">
                    <Text className="text-xl font-psemibold mb-3 text-gray-800">Ride Options</Text>
                    <View className="flex-row justify-between">
                      {rideOptions.map((option) => (
                        <TouchableHighlight
                          key={option.id}
                          className="flex-1 mx-2 rounded-xl overflow-hidden"
                          underlayColor="#E5E7EB"
                          onPress={() => { }}
                        >
                          <View
                            className="items-center justify-center p-4"
                            style={{ backgroundColor: "#F3F4F6" }}
                          >
                            <FontAwesome5 name={option.icon} size={24} color="#065F46" />
                            <Text className="mt-2 font-pmedium text-[#065F46]">{option.name}</Text>
                          </View>
                        </TouchableHighlight>
                      ))}
                    </View>
                  </View>
                )}

                {index === 4 && (
                  <View className="mt-8 mb-8">
                    <Text className="text-xl font-psemibold mb-3 text-gray-800">Insights</Text>
                    <FlatList
                      ref={carouselRef}
                      data={carouselData}
                      horizontal
                      showsHorizontalScrollIndicator={false}
                      keyExtractor={(item) => item.id}
                      snapToAlignment="center"
                      snapToInterval={width * 0.8 + 16}
                      decelerationRate="fast"
                      renderItem={({ item }) => (
                        <View
                          className="bg-gray-100 rounded-xl p-5 mr-4 shadow-sm border border-gray-200"
                          style={{
                            width: width * 0.8,
                            marginRight: 16,
                            borderRadius: 16,
                            elevation: 2,
                            shadowOpacity: 0.1,
                            shadowRadius: 4,
                            shadowOffset: { width: 0, height: 2 },
                          }}
                        >
                          <View className="flex-row items-center mb-4">
                            <View
                              className="bg-white p-3 rounded-full"
                              style={{ shadowOpacity: 0.1, shadowRadius: 2 }}
                            >
                              <FontAwesome5 name={item.icon} size={24} color="#065F46" />
                            </View>
                            <Text className="ml-4 text-lg font-psemibold text-gray-800">
                              {item.title}
                            </Text>
                          </View>
                          <Text className="text-base text-gray-700 mb-5 font-pregular leading-5">
                            {item.description}
                          </Text>
                          <TouchableOpacity
                            className="bg-white rounded-full py-2 px-6 self-start border border-[#065F46]"
                            style={{ shadowOpacity: 0.05, shadowRadius: 2, shadowOffset: { width: 0, height: 1 } }}
                          >
                            <Text className="text-[#065F46] font-psemibold text-sm">
                              {item.buttonText}
                            </Text>
                          </TouchableOpacity>
                        </View>
                      )}
                      contentContainerStyle={{
                        paddingHorizontal: 8,
                      }}
                    />
                    <View className="flex-row justify-center mt-3">
                      {carouselData.map((_, idx) => (
                        <View
                          key={idx}
                          className={`h-1.5 w-1.5 rounded-full mx-1 ${idx === 0 ? 'bg-[#065F46]' : 'bg-gray-300'}`}
                        />
                      ))}
                    </View>
                  </View>
                )}
              </View>
            );
          }}
          showsVerticalScrollIndicator={false}
        />
      </LinearGradient>

      <Modal
        animationType="slide"
        transparent={true}
        visible={showRecordingModal}
        onRequestClose={() => {
          if (isRecording) {
            alert("Please stop recording before closing.");
          } else {
            setShowRecordingModal(false);
          }
        }}
      >
        <Pressable
          className="flex-1 justify-end items-center"
          style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
          onPress={() => {
            if (!isRecording && status !== "transcribing") {
              setShowRecordingModal(false);
            }
          }}
        >
          <Animated.View
            entering={FadeInUp.duration(400)}
            style={{
              width: '100%',
              height: height * 0.35,
              backgroundColor: 'white',
              borderTopLeftRadius: 24,
              borderTopRightRadius: 24,
              shadowColor: '#000',
              shadowOffset: { width: 0, height: -4 },
              shadowOpacity: 0.25,
              shadowRadius: 12,
              elevation: 25,
              padding: 20,
            }}
          >
            <View style={{ alignItems: 'center', marginBottom: 16 }}>
              <View
                style={{
                  width: 40,
                  height: 4,
                  backgroundColor: '#E5E7EB',
                  borderRadius: 2,
                }}
              />
            </View>

            <View style={{ flex: 1, justifyContent: 'space-between' }}>
              <View style={{ flex: 1, marginBottom: 16 }}>
                <TextInput
                  style={{
                    flex: 1,
                    backgroundColor: '#F3F4F6',
                    borderRadius: 16,
                    paddingHorizontal: 16,
                    paddingVertical: 12,
                    fontSize: 16,
                    fontFamily: 'pregular',
                    color: '#374151',
                    textAlignVertical: 'top',
                    borderWidth: 1,
                    borderColor: '#E5E7EB',
                  }}
                  placeholder="Speak or type your query..."
                  placeholderTextColor="#9CA3AF"
                  value={textInputContent}
                  onChangeText={setTextInputContent}
                  editable={!isRecording && status !== "transcribing"}
                  multiline={true}
                  numberOfLines={6}
                  maxLength={1000}
                />
                
                <Text style={{
                  fontSize: 12,
                  color: '#9CA3AF',
                  textAlign: 'right',
                  marginTop: 4
                }}>
                  {textInputContent.length}/1000
                </Text>
              </View>

              <View style={{
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 16,
              }}>
                <TouchableOpacity
                  onPress={isRecording ? stopRecording : startRecording}
                  disabled={status === "requesting permissions" || status === "transcribing"}
                  style={{
                    width: 56,
                    height: 56,
                    borderRadius: 28,
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: status === "transcribing" ? "#047857" :
                                  isRecording ? "#DC2626" : "#047857",
                    elevation: 6,
                    shadowColor: isRecording ? "#DC2626" : "#047857",
                    shadowOffset: { width: 0, height: 3 },
                    shadowOpacity: 0.3,
                    shadowRadius: 4,
                  }}
                  activeOpacity={0.8}
                >
                  {status === "transcribing" ? (
                    <ActivityIndicator size="small" color="white" />
                  ) : (
                    <FontAwesome5
                      name={isRecording ? "stop" : "microphone"}
                      size={22}
                      color="white"
                    />
                  )}
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={handleSendMessage}
                  disabled={!textInputContent.trim() || isRecording || status === "transcribing"}
                  style={{
                    width: 56,
                    height: 56,
                    borderRadius: 28,
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: (!textInputContent.trim() || isRecording || status === "transcribing") 
                      ? "#9CA3AF" : "#047857",
                    elevation: 6,
                    shadowColor: "#047857",
                    shadowOffset: { width: 0, height: 3 },
                    shadowOpacity: 0.3,
                    shadowRadius: 4,
                  }}
                  activeOpacity={0.8}
                >
                  <FontAwesome5 name="paper-plane" size={22} color="white" />
                </TouchableOpacity>
              </View>
            </View>
          </Animated.View>
        </Pressable>
      </Modal>
    </SafeAreaView>
  );
};

export default Home;