import "./App.css";
import { useState } from "react";
import DrawingPanel from "./components/Drawing/DrawingPanel";
import Header from "./components/Layout/Header";
import useModels from "./hooks/use-models";
import Results from "./components/Layout/Results";

function App() {
  const [outputData, setOuputData] = useState([]);
  const [outputIdx, setOutputIdx] = useState(0);

  const [modelsData, loadModel] = useModels("http://127.0.0.1:8000/api/models");

  const processOutputData = (data) => {
    setOuputData(data);
    setOutputIdx((idx) => idx + 1);
  };

  return (
    <div className="App">
      <Header modelsData={modelsData} onButtonClick={loadModel} />

      <DrawingPanel
        className="drawing-panel"
        modelsData={modelsData}
        onOutput={processOutputData}
      />

      <Results
        outputData={outputData}
        outputIdx={outputIdx}
        modelsData={modelsData}
      />
    </div>
  );
}

export default App;
