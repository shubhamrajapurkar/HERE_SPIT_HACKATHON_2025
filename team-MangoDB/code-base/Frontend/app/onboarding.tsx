import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, ScrollView, Animated } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useForm, Controller, Control, FieldErrors, useWatch } from "react-hook-form";
import { router } from "expo-router";

interface FormData {
  workStartTime: string;
  officeFrequency: string;
  homeAddress: string;
  workAddress: string;
  preferredTransport: string;
  carpoolingInterest: string;
}

interface StepProps {
  control: Control<FormData>;
  errors: FieldErrors<FormData>;
}

const OnboardingScreen = () => {
  const [step, setStep] = useState(1);
  const { control, handleSubmit, formState: { errors } } = useForm<FormData>();
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true
    }).start();
  }, [step]);

  const onSubmit = (data: FormData) => {
    console.log(data);
    // Send this data to the backend
    router.push('/home');
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return <Step1 control={control} errors={errors} />;
      case 2:
        return <Step2 control={control} errors={errors} />;
      case 3:
        return <Step3 control={control} errors={errors} />;
      default:
        return null;
    }
  };

  return (
    <SafeAreaView className="flex-1 bg-[#ebf4fa]">
      <ScrollView className="flex-1 px-6 py-10">
        <Animated.View style={{ opacity: fadeAnim }}>
          {renderStep()}
        </Animated.View>
        <View className="flex-row justify-between mt-6">
          {step > 1 && (
            <TouchableOpacity
              onPress={() => setStep(step - 1)}
              className="bg-white border-2 border-emerald-600 py-3 px-8 rounded-full shadow-md"
            >
              <Text className="text-emerald-600 font-semibold">Previous</Text>
            </TouchableOpacity>
          )}
          {step < 3 ? (
            <TouchableOpacity
              onPress={() => setStep(step + 1)}
              className="bg-white border-2 border-emerald-600 py-3 px-8 rounded-full ml-auto shadow-lg"
            >
              <Text className="text-emerald-600 font-semibold">Next</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              onPress={handleSubmit(onSubmit)}
              className="bg-white border-2 border-emerald-600 py-3 px-8 rounded-full ml-auto shadow-lg"
            >
              <Text className="text-emerald-600 font-semibold">Finish</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const Step1: React.FC<StepProps> = ({ control, errors }) => {
  const officeFrequency = useWatch({
    control,
    name: "officeFrequency"
  });

  return (
    <View className="bg-white rounded-2xl p-6 shadow-lg">
      <Text className="text-2xl font-semibold text-primary mb-6">When do you start your day? 🌅</Text>
      <Controller
        control={control}
        rules={{ required: true }}
        render={({ field: { onChange, onBlur, value } }) => (
          <TextInput
            placeholder="What time do you start work?"
            onBlur={onBlur}
            onChangeText={onChange}
            value={value}
            className="bg-gray-50 rounded-xl p-4 mb-4 border border-emerald-100"
          />
        )}
        name="workStartTime"
      />
      {errors.workStartTime && <Text className="text-red-500 mb-4">This field is required.</Text>}
      <Text className="text-xl font-medium text-primary mb-4">How often do you visit office? 🏢</Text>
      {["Daily", "5 times a week", "3-4 times a week", "1-2 times a week"].map((option) => (
        <Controller
          key={option}
          control={control}
          name="officeFrequency"
          render={({ field: { onChange } }) => (
            <TouchableOpacity
              onPress={() => onChange(option)}
              className={`p-4 rounded-xl mb-3 ${
                officeFrequency === option ? "bg-primary" : "bg-gray-50"
              }`}
            >
              <Text className={`${officeFrequency === option ? "text-white" : "text-primary"}`}>
                {option}
              </Text>
            </TouchableOpacity>
          )}
        />
      ))}
      {errors.officeFrequency && <Text className="text-red-500 mb-4">This field is required.</Text>}
    </View>
  );
};

const Step2: React.FC<StepProps> = ({ control, errors }) => (
  <View className="bg-white rounded-2xl p-6 shadow-lg">
    <Text className="text-2xl font-semibold text-primary mb-6">Where's your daily journey? 🗺️</Text>
    <Controller
      control={control}
      rules={{ required: true }}
      render={({ field: { onChange, onBlur, value } }) => (
        <TextInput
          placeholder="What's your home address?"
          onBlur={onBlur}
          onChangeText={onChange}
          value={value}
          className="bg-gray-50 rounded-xl p-4 mb-4 border border-emerald-100"
        />
      )}
      name="homeAddress"
    />
    {errors.homeAddress && <Text className="text-red-500 mb-4">This field is required.</Text>}
    <Controller
      control={control}
      rules={{ required: true }}
      render={({ field: { onChange, onBlur, value } }) => (
        <TextInput
          placeholder="What's your work address?"
          onBlur={onBlur}
          onChangeText={onChange}
          value={value}
          className="bg-gray-50 rounded-xl p-4 mb-4 border border-emerald-100"
        />
      )}
      name="workAddress"
    />
    {errors.workAddress && <Text className="text-red-500 mb-4">This field is required.</Text>}
  </View>
);

const Step3: React.FC<StepProps> = ({ control, errors }) => {
  const preferredTransport = useWatch({
    control,
    name: "preferredTransport"
  });
  
  const carpoolingInterest = useWatch({
    control,
    name: "carpoolingInterest"
  });

  return (
    <View className="bg-white rounded-2xl p-6 shadow-lg">
      <Text className="text-2xl font-semibold text-primary mb-6">How do you like to travel? 🚗</Text>
      {["Public Transport", "Car", "Bike", "Walk"].map((option) => (
        <Controller
          key={option}
          control={control}
          name="preferredTransport"
          render={({ field: { onChange } }) => (
            <TouchableOpacity
              onPress={() => onChange(option)}
              className={`p-4 rounded-xl mb-3 shadow-sm ${
                preferredTransport === option ? "bg-primary" : "bg-white"
              }`}
            >
              <Text className={`text-lg ${preferredTransport === option ? "text-white" : "text-primary"}`}>
                {option}
              </Text>
            </TouchableOpacity>
          )}
        />
      ))}
      {errors.preferredTransport && <Text className="text-red-500 mb-4">This field is required.</Text>}
      {/* <Text className="text-xl font-medium text-primary my-4">Want to share rides? 🚗💨</Text>
      {["Yes", "No", "Maybe"].map((option) => (
        <Controller
          key={option}
          control={control}
          name="carpoolingInterest"
          render={({ field: { onChange } }) => (
            <TouchableOpacity
              onPress={() => onChange(option)}
              className={`p-4 rounded-xl mb-3 shadow-sm ${
                carpoolingInterest === option ? "bg-primary" : "bg-gray-50"
              }`}
            >
              <Text className={`text-lg ${carpoolingInterest === option ? "text-white" : "text-primary"}`}>
                {option}
              </Text>
            </TouchableOpacity>
          )}
        />
      ))}
      {errors.carpoolingInterest && <Text className="text-red-500 mb-4">This field is required.</Text>} */}
    </View>
  );
};

export default OnboardingScreen;
