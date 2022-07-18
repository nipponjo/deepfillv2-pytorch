import React from "react";
import classes from "./ModelTab.module.css";

const ModelTab = (props) => {
  const tabStyle = `${classes.tab} ${props.data.is_loaded ? classes['loaded'] : classes['not-loaded']}`
  return (
    <div className={tabStyle}>
      {`${props.data.type}: ${props.data.name}`}
      <button
        onClick={() => {
          props.onButtonClick(props.data.is_loaded, props.data.name);
        }}
      >
        {props.data.is_loaded ? "X" : "+"}
      </button>
    </div>
  );
};

export default ModelTab;
