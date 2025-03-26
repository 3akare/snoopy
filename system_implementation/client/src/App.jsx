import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:8080/", { query });
      setResponse(res.data);
    } catch (error) {
      console.error("Error:", error);
      setResponse("Error occurred");
    }
  };

  return (
    <div>
      <h1>Send a Request</h1>
      <textarea
        rows="4"
        placeholder="Enter your text..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSubmit}> Send </button>
      {response && (
        <div>
          <strong>Response:</strong> {response}
        </div>
      )}
    </div>
  );
}

export default App;
