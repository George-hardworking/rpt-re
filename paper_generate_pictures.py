# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import os.path as op
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from Data import equity_data as eqd
from Data import chart_library as cl
from Data import dgp_config as dcf
from Misc import utilities as ut


class ChartGenerationError(Exception):
    pass


class GenerateStockData(object):
    def __init__(
        self,
        country,
        year,
        window_size,
        freq,
        chart_freq=1,
        ma_lags=None,
        volume_bar=False,
        need_adjust_price=True,
        allow_tqdm=True,
        chart_type="bar",
    ):
        self.country = country
        self.year = year
        self.window_size = window_size
        self.freq = freq
        assert self.freq in ["week", "month", "quarter"]
        self.chart_freq = chart_freq
        assert window_size % chart_freq == 0
        self.chart_len = int(window_size / chart_freq)
        assert self.chart_len in [5, 20, 60]
        self.ma_lags = ma_lags
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.allow_tqdm = allow_tqdm
        assert chart_type in ["bar", "pixel", "centered_pixel"]
        self.chart_type = chart_type

        self.ret_len_list = [5, 20, 60, 65, 180, 250, 260]
        self.bar_width = 3
        self.image_width = {
            5: self.bar_width * 5,
            20: self.bar_width * 20,
            60: self.bar_width * 60,
        }
        self.image_height = {5: 32, 20: 64, 60: 96}

        self.width, self.height = (
            self.image_width[int(self.chart_len)],
            self.image_height[int(self.chart_len)],
        )

        self.df = None
        self.stock_id_list = None

        self.save_dir = ut.get_dir(
            op.join(dcf.STOCKS_SAVEPATH, f"stocks_{self.country}/dataset_all")
        )
        self.image_save_dir = ut.get_dir(op.join(dcf.STOCKS_SAVEPATH, "sample_images"))
        vb_str = "has_vb" if self.volume_bar else "no_vb"
        ohlc_len_str = "" if self.chart_freq == 1 else f"_{self.chart_len}ohlc"
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        self.file_name = f"{chart_type_str}{self.window_size}d_{self.freq}_{vb_str}_{str(self.ma_lags)}_ma_{self.year}{ohlc_len_str}"
        self.log_file_name = op.join(self.save_dir, f"{self.file_name}.txt")
        self.labels_filename = op.join(
            self.save_dir, f"{self.file_name}_labels.feather"
        )
        self.images_filename = op.join(self.save_dir, f"{self.file_name}_images.dat")

    @staticmethod
    def adjust_price(df):
        if len(df) == 0:
            raise ChartGenerationError("adjust_price: Empty Dataframe")
        if len(df.Date.unique()) != len(df):
            raise ChartGenerationError("adjust_price: Dates not unique")
        df = df.reset_index(drop=True)

        fd_close = abs(df.at[0, "Close"])
        if df.at[0, "Close"] == 0.0 or pd.isna(df.at[0, "Close"]):
            raise ChartGenerationError("adjust_price: First day close is nan or zero")

        pre_close = fd_close
        res_df = df.copy()

        res_df.at[0, "Close"] = 1.0
        res_df.at[0, "Open"] = abs(res_df.at[0, "Open"]) / pre_close
        res_df.at[0, "High"] = abs(res_df.at[0, "High"]) / pre_close
        res_df.at[0, "Low"] = abs(res_df.at[0, "Low"]) / pre_close

        pre_close = 1
        for i in range(1, len(res_df)):
            today_closep = abs(res_df.at[i, "Close"])
            today_openp = abs(res_df.at[i, "Open"])
            today_highp = abs(res_df.at[i, "High"])
            today_lowp = abs(res_df.at[i, "Low"])
            today_ret = np.float64(res_df.at[i, "Ret"])

            res_df.at[i, "Close"] = (1 + today_ret) * pre_close
            res_df.at[i, "Open"] = res_df.at[i, "Close"] / today_closep * today_openp
            res_df.at[i, "High"] = res_df.at[i, "Close"] / today_closep * today_highp
            res_df.at[i, "Low"] = res_df.at[i, "Close"] / today_closep * today_lowp
            res_df.at[i, "Ret"] = today_ret

            if not pd.isna(res_df.at[i, "Close"]):
                pre_close = res_df.at[i, "Close"]

        return res_df

    def load_adjusted_daily_prices(self, stock_df, date):
        if date not in set(stock_df.Date):
            return 0
        date_index = stock_df[stock_df.Date == date].index[0]
        ma_offset = 0 if self.ma_lags is None else np.max(self.ma_lags)
        data = stock_df.loc[
            (date_index - (self.window_size - 1) - ma_offset) : date_index
        ]
        if len(data) < self.window_size:
            return 1
        if len(data) < (self.window_size + ma_offset):
            ma_lags = []
            data = stock_df.loc[(date_index - (self.window_size - 1)) : date_index]
        else:
            ma_lags = self.ma_lags
        if self.chart_freq != 1:
            data = self.convert_daily_df_to_chart_freq_df(data)
        if self.need_adjust_price:
            if data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0]):
                return 2
            data = self.adjust_price(data)
        else:
            data = data.copy()
        start_date_index = data.index[-1] - self.chart_len + 1
        if data["Close"].loc[start_date_index] == 0 or np.isnan(
            data["Close"].loc[start_date_index]
        ):
            return 3
        data[["Open", "High", "Low", "Close"]] *= (
            1.0 / data["Close"].loc[start_date_index]
        )
        if self.ma_lags is not None:
            ma_name_list = ["ma" + str(lag) for lag in ma_lags]
            for i, ma_name in enumerate(ma_name_list):
                chart_num = int(ma_lags[i] / self.chart_freq)
                data[ma_name] = (
                    data["Close"].rolling(chart_num, win_type=None).sum() / chart_num
                )

        data["Prev_Close"] = data["Close"].shift(1)

        df = data.loc[start_date_index:]
        if (
            len(df) != self.chart_len
            or np.around(df.iloc[0]["Close"], decimals=3) != 1.000
        ):
            return 4
        df = df.reset_index(drop=True)
        return df

    def convert_daily_df_to_chart_freq_df(self, daily_df):
        if not len(daily_df) % self.chart_freq == 0:
            raise ChartGenerationError("df not divided by chart freq")
        ohlc_len = int(len(daily_df) / self.chart_freq)
        df = pd.DataFrame(index=range(ohlc_len), columns=daily_df.columns)
        for i in range(ohlc_len):
            subdata = daily_df.iloc[
                int(i * self.chart_freq) : int((i + 1) * self.chart_freq)
            ]
            df.loc[i] = subdata.iloc[-1]
            df.loc[i, "Open"] = subdata.iloc[0]["Open"]
            df.loc[i, "High"] = subdata["High"].max()
            df.loc[i, "Low"] = subdata["Low"].min()
            df.loc[i, "Vol"] = subdata["Vol"].sum()
            df.loc[i, "Ret"] = np.prod(1 + np.array(subdata["Ret"])) - 1
        return df

    def _generate_daily_features(self, stock_df, date):
        res = self.load_adjusted_daily_prices(stock_df, date)
        if isinstance(res, int):
            return res
        else:
            df = res
        ma_lags = [int(ma_col[2:]) for ma_col in [c for c in df.columns if "ma" in c]]
        ohlc_obj = cl.DrawOHLC(
            df,
            has_volume_bar=self.volume_bar,
            ma_lags=ma_lags,
            chart_type=self.chart_type,
        )
        image_data = ohlc_obj.draw_image()

        if image_data is None:
            return 5

        last_day = df[df.Date == date].iloc[0]
        feature_dict = {feature: last_day[feature] for feature in stock_df.columns}
        ret_list = ["Ret"] + [f"Ret_{i}d" for i in self.ret_len_list]
        for ret in ret_list:
            feature_dict[f"{ret}_label"] = (
                1 if feature_dict[ret] > 0 else 0 if feature_dict[ret] <= 0 else 2
            )
            vol = feature_dict["EWMA_vol"]
            feature_dict[f"{ret}_tstat"] = (
                0 if (vol == 0 or pd.isna(vol)) else feature_dict[ret] / vol
            )
        feature_dict["image"] = image_data
        feature_dict["window_size"] = self.window_size
        feature_dict["Date"] = date
        return feature_dict

    def _get_feature_and_dtype_list(self):

        float32_features = (
            [
                "EWMA_vol",
                "Ret",
                "Ret_tstat",
                "Ret_week",
                "Ret_month",
                "Ret_quarter",
                "Ret_year",
                "MarketCap",
            ]
            + [f"Ret_{i}d" for i in self.ret_len_list]
            + [f"Ret_{i}d_tstat" for i in self.ret_len_list]
        )
        int8_features = ["Ret_label"] + [f"Ret_{i}d_label" for i in self.ret_len_list]
        uint8_features = ["image", "window_size"]
        object_features = ["StockID"]
        datetime_features = ["Date"]
        feature_list = (
            float32_features
            + int8_features
            + uint8_features
            + object_features
            + datetime_features
        )
        float32_dict = {feature: np.float32 for feature in float32_features}
        int8_dict = {feature: np.int8 for feature in int8_features}
        uint8_dict = {feature: np.uint8 for feature in uint8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        dtype_dict = {
            **float32_dict,
            **int8_dict,
            **uint8_dict,
            **object_dict,
            **datetime_dict,
        }
        return dtype_dict, feature_list

    def save_annual_data(self):
        if (
            op.isfile(self.log_file_name)
            and op.isfile(self.labels_filename)
            and op.isfile(self.images_filename)
        ):
            print("Found pregenerated file {}".format(self.file_name))
            return
        print(f"Generating {self.file_name}")
        self.df = eqd.get_processed_US_data_by_year(self.year)
        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        data_miss = np.zeros(6)
        data_dict = {
            feature: np.empty(len(self.stock_id_list) * 60, dtype=dtype_dict[feature])
            for feature in feature_list
        }
        data_dict["image"] = np.empty(
            [len(self.stock_id_list) * 60, self.width * self.height],
            dtype=dtype_dict["image"],
        )
        data_dict["image"].fill(np.nan)

        sample_num = 0
        iterator = (
            tqdm(self.stock_id_list)
            if (self.allow_tqdm and "tqdm" in sys.modules)
            else self.stock_id_list
        )
        for i, stock_id in enumerate(iterator):
            stock_df = self.df.xs(stock_id, level=1).copy()
            stock_df = stock_df.reset_index()
            dates = eqd.get_period_end_dates(self.freq)
            dates = dates[dates.year == self.year]
            for j, date in enumerate(dates):
                try:
                    image_label_data = self._generate_daily_features(stock_df, date)

                    if type(image_label_data) is dict:
                        if (i < 2) and (j == 0):
                            image_label_data["image"].save(
                                op.join(
                                    self.image_save_dir,
                                    f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png",
                                )
                            )

                        image_label_data["StockID"] = stock_id
                        im_arr = np.frombuffer(
                            image_label_data["image"].tobytes(), dtype=np.uint8
                        )
                        assert im_arr.size == self.width * self.height
                        data_dict["image"][sample_num, :] = im_arr[:]
                        for feature in [x for x in feature_list if x != "image"]:
                            data_dict[feature][sample_num] = image_label_data[feature]
                        sample_num += 1
                    elif type(image_label_data) is int:
                        data_miss[image_label_data] += 1
                    else:
                        raise ValueError
                except ChartGenerationError:
                    print(f"DGP Error on {stock_id} {date}")
                    continue
        for feature in feature_list:
            data_dict[feature] = data_dict[feature][:sample_num]

        fp_x = np.memmap(
            self.images_filename,
            dtype=np.uint8,
            mode="w+",
            shape=data_dict["image"].shape,
        )
        fp_x[:] = data_dict["image"][:]
        del fp_x
        print(f"Save image data to {self.images_filename}")
        data_dict = {x: data_dict[x] for x in data_dict.keys() if x != "image"}
        pd.DataFrame(data_dict).to_feather(self.labels_filename)
        print(f"Save label data to {self.labels_filename}")
        log_file = open(self.log_file_name, "w+")
        log_file.write(
            "total_dates:%d total_missing:%d type0:%d type1:%d type2:%d type3:%d type4:%d type5:%d"
            % (
                sample_num,
                sum(data_miss),
                data_miss[0],
                data_miss[1],
                data_miss[2],
                data_miss[3],
                data_miss[4],
                data_miss[5],
            )
        )
        log_file.close()
        print(f"Save log file to {self.log_file_name}")

    def generate_daily_ts_features(self, stock_df, date):
        if date not in set(stock_df.Date):
            return 0

        date_index = stock_df[stock_df.Date == date].index[0]

        ma_offset = 0 if self.ma_lags is None else np.max(self.ma_lags) - 1
        data = stock_df.loc[
            (date_index - (self.window_size - 1) - ma_offset) : date_index
        ]

        if len(data) != (self.window_size + ma_offset):
            return 1

        if data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0]):
            return 2
        data = self.adjust_price(data)

        start_date_index = data.index[-1] - self.window_size + 1
        if data["Close"].loc[start_date_index] == 0 or np.isnan(
            data["Close"].loc[start_date_index]
        ):
            return 3

        data[["Open", "High", "Low", "Close"]] *= (
            1.0 / data["Close"].loc[start_date_index]
        )

        if self.ma_lags is not None:
            ma_name_list = ["ma" + str(lag) for lag in self.ma_lags]
            for i, ma_name in enumerate(ma_name_list):
                data[ma_name] = (
                    data["Close"].rolling(self.ma_lags[i], win_type=None).sum()
                    / self.ma_lags[i]
                )

        window = data.loc[start_date_index:]

        if (
            len(window) != self.window_size
            or np.around(window.iloc[0]["Close"], decimals=3) != 1.000
        ):
            return 4

        window = window.reset_index(drop=True)

        predictors = np.zeros(shape=(6, self.window_size))
        predictors = window[
            ["Open", "High", "Low", "Close", "ma" + str(self.window_size), "Vol"]
        ].T.to_numpy()

        last_day = data[data.Date == date]
        assert len(last_day) == 1
        last_day = last_day.iloc[0]

        feature_list = [
            "StockID",
            "Date",
            "EWMA_vol",
            "Ret",
            "Ret_5d",
            "Ret_20d",
            "Ret_60d",
            "Ret_week",
            "Ret_month",
            "Ret_quarter",
            "MarketCap",
        ]
        feature_dict = {feature: last_day[feature] for feature in feature_list}

        ret_list = ["Ret", "Ret_5d", "Ret_20d", "Ret_60d"]
        for ret in ret_list:
            feature_dict["{}_label".format(ret)] = (
                1 if feature_dict[ret] > 0 else 0 if feature_dict[ret] <= 0 else 2
            )
            vol = feature_dict["EWMA_vol"]
            feature_dict["{}_tstat".format(ret)] = (
                0 if (vol == 0 or pd.isna(vol)) else feature_dict[ret] / vol
            )
        feature_dict["predictor"] = predictors
        feature_dict["window_size"] = self.window_size

        return feature_dict

    def get_ts_feature_and_dtype_list(self):
        float32_features = [
            "EWMA_vol",
            "Ret",
            "Ret_5d",
            "Ret_20d",
            "Ret_60d",
            "Ret_week",
            "Ret_month",
            "Ret_quarter",
            "Ret_tstat",
            "Ret_5d_tstat",
            "Ret_20d_tstat",
            "Ret_60d_tstat",
            "MarketCap",
            "predictor",
        ]
        int8_features = [
            "Ret_label",
            "Ret_5d_label",
            "Ret_20d_label",
            "Ret_60d_label",
            "window_size",
        ]
        object_features = ["StockID"]
        datetime_features = ["Date"]
        feature_list = (
            float32_features + int8_features + object_features + datetime_features
        )
        float32_dict = {feature: np.float32 for feature in float32_features}
        int8_dict = {feature: np.int8 for feature in int8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        dtype_dict = {**float32_dict, **int8_dict, **object_dict, **datetime_dict}
        return dtype_dict, feature_list

    def save_annual_ts_data(self):
        dtype_dict, feature_list = self.get_ts_feature_and_dtype_list()
        file_name = "{}d_{}_{}_vb_{}_ma_{}_ts".format(
            self.window_size,
            self.freq,
            "has" if self.volume_bar else "no",
            str(self.ma_lags),
            self.year,
        )
        log_file_name = os.path.join(self.save_dir, "{}.txt".format(file_name))
        data_filename = os.path.join(self.save_dir, "{}_data_new.npz".format(file_name))
        if os.path.isfile(log_file_name) and os.path.isfile(data_filename):
            print("Found pregenerated file {}".format(file_name))
            return

        data_miss = np.zeros(6)
        data_dict = {
            feature: np.empty(len(self.stock_id_list) * 60, dtype=dtype_dict[feature])
            for feature in feature_list
        }
        data_dict["predictor"] = np.empty(
            [len(self.stock_id_list) * 60, 6, self.window_size], dtype=np.float32
        )
        data_dict["predictor"].fill(np.nan)

        sample_num = 0
        iterator = (
            tqdm(self.stock_id_list)
            if (self.allow_tqdm and "tqdm" in sys.modules)
            else self.stock_id_list
        )
        for i, stock_id in enumerate(iterator):
            df = self.df[self.df.StockID == stock_id]
            df = df.reset_index(drop=True)
            dates = df[~pd.isna(df["Ret_{}".format(self.freq)])].Date
            dates = dates[dates.dt.year == self.year]
            for j, date in enumerate(dates):
                try:
                    predictor_label_data = self.generate_daily_ts_features(df, date)
                    if type(predictor_label_data) is dict:
                        for feature in feature_list:
                            data_dict[feature][sample_num] = predictor_label_data[
                                feature
                            ]
                        sample_num += 1
                    elif type(predictor_label_data) is int:
                        data_miss[predictor_label_data] += 1
                    else:
                        raise ValueError
                except ChartGenerationError:
                    continue
        for feature in feature_list:
            data_dict[feature] = data_dict[feature][:sample_num]

        np.savez_compressed(
            data_filename, data_dict={x: data_dict[x] for x in data_dict.keys()}
        )
        log_file = open(log_file_name, "w+")
        log_file.write(
            "total_dates:%d total_missing:%d type0:%d type1:%d type2:%d type3:%d type4:%d type5:%d"
            % (
                sample_num,
                sum(data_miss),
                data_miss[0],
                data_miss[1],
                data_miss[2],
                data_miss[3],
                data_miss[4],
                data_miss[5],
            )
        )
        log_file.close()


def main():
    pass


if __name__ == "__main__":
    main()
