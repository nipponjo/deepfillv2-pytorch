import React from "react";
import classes from "./Header.module.css";

import ModelTab from "../UI/ModelTab";

const Header = (props) => {
  if (!props.modelsData) {
    return <pre>Loading models...</pre>;
  }

  const loadedModels = props.modelsData.filter((model) => model.is_loaded);
  const unloadedModels = props.modelsData.filter((model) => !model.is_loaded);
  const sortedModels = loadedModels.concat(unloadedModels);

  return (
    <header className={classes.header}>
      <div className={classes["model-select"]}>
        {sortedModels.map((modelData) => (
          <ModelTab
            key={modelData.name}
            data={modelData}
            onButtonClick={props.onButtonClick}
          />
        ))}
      </div>
    </header>
  );
};

export default Header;
