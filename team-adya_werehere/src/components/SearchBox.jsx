import { useState, useEffect, useCallback } from "react";
import {
	Combobox,
	ComboboxInput,
	ComboboxPopover,
	ComboboxList,
	ComboboxOption,
} from "@reach/combobox";
import "@reach/combobox/styles.css";

const SearchBox = ({ setPosition, position }) => {
	const [value, setValue] = useState("");
	const [suggestions, setSuggestions] = useState([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [isListening, setIsListening] = useState(false);
	const [isProcessing, setIsProcessing] = useState(false);
	const [recognition, setRecognition] = useState(null);

	const initializeSpeechRecognition = useCallback(() => {
		if (typeof window === 'undefined') return;

		const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
		if (!SpeechRecognition) {
			setError("Speech recognition is not supported in your browser");
			return;
		}

		const recognition = new SpeechRecognition();
		recognition.continuous = false;
		recognition.interimResults = true;
		recognition.maxAlternatives = 3;
		recognition.lang = 'en-US';

		recognition.onstart = () => {
			setIsListening(true);
			setIsProcessing(true);
			setError(null);
		};

		recognition.onresult = (event) => {
			const transcript = Array.from(event.results)
				.map(result => result[0].transcript)
				.join('')
				.toLowerCase()
				.trim();

			if (transcript.length > 0) {
				setValue(transcript);
				searchPlaces(transcript);
			}
		};

		recognition.onerror = (event) => {
			console.error('Speech recognition error:', event.error);
			switch (event.error) {
				case 'no-speech':
					setError("No speech was detected. Please try again.");
					break;
				case 'audio-capture':
					setError("No microphone was found. Please ensure your microphone is connected.");
					break;
				case 'not-allowed':
					setError("Microphone access was denied. Please allow microphone access.");
					break;
				default:
					setError("An error occurred with speech recognition. Please try again.");
			}
			setIsListening(false);
			setIsProcessing(false);
		};

		recognition.onend = () => {
			setIsListening(false);
			setIsProcessing(false);
		};

		setRecognition(recognition);
	}, []);

	useEffect(() => {
		initializeSpeechRecognition();
		return () => {
			if (recognition) {
				recognition.stop();
			}
		};
	}, [initializeSpeechRecognition]);

	const startListening = useCallback(() => {
		if (!recognition) {
			setError("Speech recognition is not available");
			return;
		}

		try {
			recognition.start();
		} catch (error) {
			console.error('Error starting speech recognition:', error);
			setError("Failed to start voice search. Please try again.");
		}
	}, [recognition]);

	const searchPlaces = async (query) => {
		if (!query) {
			setSuggestions([]);
			return;
		}

		setLoading(true);
		setError(null);

		const storedPreferences = JSON.parse(localStorage.getItem('userPreferences'));
		const dietaryPreferences = storedPreferences?.diet_type?.reduce((acc, pref) => {
			const prefMap = {
				'vegetarian': 'vegetarian',
				'vegan': 'vegan',
				'halal': 'halal',
				'kosher': 'kosher',
				'gluten-free': 'gluten_free',
				'jain': 'jain'
			};
			if (prefMap[pref]) {
				acc[prefMap[pref]] = true;
			}
			return acc;
		}, {});

		try {
			const response = await fetch('http://localhost:8000/api/v1/search', {
				method: 'POST',
				headers: {
					'Accept': 'application/json',
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					query: query,
					dietary_preferences: dietaryPreferences
				})
			});

			if (!response.ok) {
				const errorText = await response.text();
				console.error('Server error:', errorText);
				throw new Error('Failed to fetch results');
			}

			const data = await response.json();
			setSuggestions(data.results || []);
		} catch (error) {
			console.error("Error in search:", error);
			setError("Failed to search places. Please try again.");
			setSuggestions([]);
		} finally {
			setLoading(false);
		}
	};

	const handleInput = (e) => {
		const newValue = e.target.value;
		setValue(newValue);
		if (newValue.length >= 2) {
			searchPlaces(newValue);
		} else {
			setSuggestions([]);
		}
	};

	const handleSelect = async (name) => {
		setValue(name);
		const selected = suggestions.find((item) => item.name === name);
		if (selected && selected.coords) {
			setPosition({
				lat: selected.coords.lat,
				lng: selected.coords.lng
			});
			setSuggestions([]);
		}
	};

	return (
		<div className="relative w-full">
			<div className="flex items-center">
				<Combobox onSelect={handleSelect} className="flex-grow">
					<ComboboxInput
						value={value}
						onChange={handleInput}
						className="w-full py-2 px-6 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder="Search restaurants..."
						aria-label="Search restaurants"
					/>
					{error && (
						<div className="text-red-500 text-sm mt-1">{error}</div>
					)}
					<ComboboxPopover className="z-50">
						<ComboboxList className="bg-white shadow-lg rounded-md overflow-hidden">
							{loading ? (
								<div className="p-4 text-gray-500">Loading...</div>
							) : suggestions.length > 0 ? (
								suggestions.map((restaurant, index) => (
									<ComboboxOption
										key={index}
										value={restaurant.name}
										className="p-2 hover:bg-gray-100 cursor-pointer"
									>
										<div className="flex items-center justify-between">
											<div>
												<div className="font-medium">{restaurant.name}</div>
												<div className="text-sm text-gray-600">{restaurant.location}</div>
											</div>
											<div className="flex items-center space-x-2">
												{restaurant.rating && (
													<span className="text-yellow-500">★ {restaurant.rating}</span>
												)}
												{restaurant.quality && (
													<span className="text-sm text-gray-500">Quality: {restaurant.quality}</span>
												)}
											</div>
										</div>
									</ComboboxOption>
								))
							) : value.length >= 2 ? (
								<div className="p-4 text-gray-500">No results found</div>
							) : null}
						</ComboboxList>
					</ComboboxPopover>
				</Combobox>
				<button
					onClick={startListening}
					disabled={isProcessing}
					className={`ml-2 p-2 rounded-full ${isListening ? 'bg-red-500 animate-pulse' :
						isProcessing ? 'bg-yellow-500' :
							'bg-blue-500'
						} text-white hover:bg-opacity-80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
					title={isProcessing ? "Processing..." : "Voice Search"}
				>
					<svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
					</svg>
				</button>
			</div>
		</div>
	);
};

export default SearchBox;