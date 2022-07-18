import React from "react";
import classes from "./DoubleButton.module.css";

const DoubleButton = (props) => {
  return (
    <div>
      <button className={classes["button-left"]} onClick={props.onClickLeft}>
        {props.labelLeft}
      </button>
      <button className={classes["button-right"]} onClick={props.onClickRight}>
        {props.labelRight}
      </button>
    </div>
  );
};

export default DoubleButton;
