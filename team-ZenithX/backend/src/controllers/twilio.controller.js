import twilio from 'twilio';
import { HERE_API_KEY, reverseGeocode } from '../service/geocodeService.js';
import axios from 'axios';


export async function sendMessage(req, res) {
    
const accountSid = 'SAMPLE_ACCOUNT_SID'; // Replace with your Twilio Account SID
const authToken = 'SAMPLE_AUTH_TOKEN'; // Replace with your Twilio Auth Token
const client = new twilio(accountSid, authToken);

const {message,location} = req.body;

 const response = await axios.get('https://revgeocode.search.hereapi.com/v1/revgeocode', {
        params: {
          apiKey: HERE_API_KEY,
          at: `${location.latitude},${location.longitude}`,
          lang: 'en-US'
        }
      });

// const curr=await reverseGeocode(location.lat,location.lng);
console.log(response);


client.messages
  .create({
    body: message+` at ${response.data.items[0].address.label}`,
    from: '+19788006480', // your Twilio number
    to: '+918097355046'    // recipient's phone number
  })
  .then(message => {
    
    console.log(message.sid)})
  .catch(err => console.error(err));


  return res.status(200).json({
    success: true,
    message,
  });

}
