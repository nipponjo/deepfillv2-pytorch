import { useEffect, useReducer, useCallback } from "react";
const useModels = (url = "/api/models") => {
  const modelsDataReducer = (state, action) => {
    if (action.type === "INIT") {
      return action.data;
    }

    try {
      const newState = [...state];
      const idx = newState.findIndex((d) => d.name === action.name);
      if (idx < 0) return newState;
      newState[idx].is_loaded = action.type === "LOAD" ? true : false;

      return newState;
    } catch {
      return null;
    }
  };

  const [modelsData, dispatchModelsData] = useReducer(modelsDataReducer, null);

  const fetchModels = useCallback(async () => {
    const response = await fetch(url);
    const data = await response.json();

    dispatchModelsData({ type: "INIT", data: data.data });
  }, [url]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const loadModel = (loaded, name) => {
    if (!modelsData) {
      return;
    }
    dispatchModelsData({ type: loaded ? "UNLOAD" : "LOAD", name: name });
  };

  return [modelsData, loadModel];
};

export default useModels;
