/*
 * @Author: liushijie
 * @Date: 2020-07-04 19:55:00
 * @LastEditTime: 2020-07-06 14:28:28
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dist/message.h
 */
#pragma once
#include <stdexcept>
#include <string.h>

namespace dl {
struct Message {
    enum Control {
        ReduceSum, // allreduce stage 1
        Gather,    // allreduce stage 2
        Heartbeat, // heartbeart
        Barrier, // barrier
        Report, // report ip to the root
        Empty      // default empty
    };
    Control control;
    int32_t time;
    int32_t dst_rank;
    int32_t src_rank;
    int32_t uid;
    int32_t offset;
    int32_t size;
    char   data[64];

    Message()
        : control(Empty)
        , time(-1)
        , dst_rank(-1)
        , src_rank(-1)
        , uid(-1)
        , offset(-1)
        , size(-1) {
        memset(data, 0, sizeof(char) * 64);
    }
};
}