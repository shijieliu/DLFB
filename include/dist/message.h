/*
 * @Author: liushijie
 * @Date: 2020-07-04 19:55:00
 * @LastEditTime: 2020-07-04 20:02:35
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
        Empty      // default empty
    };
    Control control;
    int32_t idx;
    int32_t dst_rank;
    int32_t src_rank;
    int32_t uid;
    int32_t offset;
    int32_t size;
    float   data[64];

    Message()
        : control(Empty)
        , idx(0)
        , dst_rank(0)
        , src_rank(0)
        , uid(0)
        , offset(0)
        , size(0) {
        memset(data, 0, sizeof(float) * 64);
    }
    Message(int32_t idx_, int32_t dst_, int32_t src_, int32_t uid_,
            int32_t offset_, int32_t size_, float *data_)
        : idx(idx_)
        , dst_rank(dst_)
        , src_rank(src_)
        , uid(uid_)
        , offset(offset_)
        , size(size_) {
        memcpy(data, data_, size);
    }
};
}