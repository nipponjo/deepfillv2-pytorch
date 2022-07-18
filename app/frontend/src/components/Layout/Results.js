import React from "react";
import ResultItem from "./ResultItem";
import classes from "./Results.module.css";

const Results = (props) => {
  return (
    <div className={classes["result-wrapper"]}>
      <div className={classes["result-container"]}>
        {props.outputData.map((modelData) => {
          return (
            <ResultItem
              key={modelData.name}
              data={modelData}
              outputIdx={props.outputIdx}
            />
          );
        })}
      </div>
    </div>
  );
};

export default Results;
